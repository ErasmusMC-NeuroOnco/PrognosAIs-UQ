import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.handlers import from_engine
from tqdm import tqdm

import prognosais.IO.constants as constants
from prognosais.IO.output import write_output_table


class Evaluator:
    """
    Evaluate a trained CSNet model on a test loader.

    This class runs inference, optionally saves predicted segmentations, and
    writes case-level prediction tables to disk. When labels are available, it
    also stores ground-truth values so that downstream scripts can compute
    classification and segmentation metrics.

    Attributes:
        model: PyTorch model to evaluate.
        test_loader: Data loader providing test batches.
        device: Device used for inference.
        post_transforms: MONAI post-processing transforms applied after
            decollating each batch.
        results_dir: Base directory where predictions and metrics are stored.
    """

    def __init__(
        self,
        model,
        test_loader,
        device,
        post_transforms,
        results_dir,
        table_extension: str,
    ):
        """
        Initialize the evaluator wrapper.

        Args:
            model: Trained model to evaluate.
            test_loader: Test data loader.
            device: Device used for inference.
            post_transforms: Post-processing transforms applied to each
                decollated sample.
            results_dir: Directory where predictions and metrics will be
                written.
            table_extension: Configured extension for generated tables.

        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.post_transforms = post_transforms
        self.results_dir = results_dir
        self.table_extension = table_extension
        self.predictions_dir = os.path.join(results_dir, constants.PREDICTIONS_DIR_NAME)
        self.metrics_dir = os.path.join(results_dir, constants.METRICS_DIR_NAME)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _register_conv_layers_hooks(self):
        """
        Register forward hooks on all Conv3d layers in the model.

        The hooks store detached CPU copies of the layer outputs so that the
        caller can inspect intermediate activation maps after inference.

        Returns:
            dict: Mapping from module name to the captured layer output.
        """

        hook_outputs = {}

        def hook_fn(name):
            def hook(module, input, output):
                hook_outputs[name] = output.detach().cpu()

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                module.register_forward_hook(hook_fn(name))

        return hook_outputs

    def _probability_map_to_nifti(self, probability_map, affine):
        """
        Convert a segmentation probability tensor into a NIfTI image.

        Args:
            probability_map: Tensor containing class probabilities with shape
                classes by x by y by z.
            affine: Affine matrix from the source image or mask.

        Returns:
            nib.Nifti1Image: Probability map stored as x by y by z by classes.
        """

        probability_array = probability_map.cpu().detach().numpy().astype(np.float32)
        probability_array = np.moveaxis(probability_array, 0, -1)
        return nib.Nifti1Image(probability_array, affine)

    def evaluate(
        self,
        save_predictions: bool,
        save_prob_map: bool = False,
        save_activation_maps: bool = False,
    ):
        """
        Run inference on the full test loader and export the results.

        The method collects class probabilities and predictions for configured tasks.
        If segmentation labels are available, it also
        exports segmentation predictions and ground-truth masks. The output is
        written as configured tables in the results directory.

        Args:
            save_predictions: If True, save predicted segmentations to NIfTI.
            save_prob_map: If True, save softmax segmentation probability maps
                to NIfTI. Maps are written as x by y by z by classes.
            save_activation_maps: If True, store Conv3d activation maps for
                each batch.

        """

        self.model.to(self.device)

        self.model.eval()

        classification_tasks = list(self.model.classification_tasks)

        cases = []

        probabilities = {task_name: [] for task_name in classification_tasks}

        predictions = {task_name: [] for task_name in classification_tasks}

        ground_truth = {task_name: [] for task_name in classification_tasks}

        labels_available = False

        if save_activation_maps:
            self.activation_maps = []
            hook_outputs = self._register_conv_layers_hooks()

        with torch.inference_mode():

            bar_val = tqdm(self.test_loader)

            for data in bar_val:

                bar_val.set_description("Inference mode")

                image = data["image"].to(self.device)

                image_affines = image.meta["original_affine"]
                if constants.CASE_ID_DATA_KEY not in data:
                    raise ValueError(
                        "Inference batch does not contain the required "
                        f"'{constants.CASE_ID_DATA_KEY}' field. Build inference "
                        "data with DataGenerator."
                    )
                raw_case_ids = data[constants.CASE_ID_DATA_KEY]
                case_ids = (
                    [raw_case_ids]
                    if isinstance(raw_case_ids, str)
                    else [str(case_id) for case_id in raw_case_ids]
                )
                if len(case_ids) != len(image):
                    raise ValueError(
                        f"Received {len(case_ids)} case IDs for a batch of "
                        f"{len(image)} images."
                    )

                pred_seg, classification_outputs = self.model(image)

                if save_activation_maps:
                    batch_copy = {
                        name: output.clone() for name, output in hook_outputs.items()
                    }
                    self.activation_maps.append(batch_copy)

                classification_probabilities = {
                    task_name: F.softmax(output, dim=1)
                    for task_name, output in classification_outputs.items()
                }

                for (
                    task_name,
                    task_probabilities,
                ) in classification_probabilities.items():
                    data[f"pred_{task_name}"] = task_probabilities

                has_segmentation = pred_seg is not None
                has_mask = "mask" in data

                if has_segmentation:
                    if save_predictions or save_prob_map:
                        save_results = [
                            os.path.join(
                                self.predictions_dir,
                                case_id,
                            )
                            for case_id in case_ids
                        ]

                        for save_dir in save_results:
                            os.makedirs(save_dir, exist_ok=True)

                    pred_seg = F.softmax(pred_seg, dim=1)
                    pred_seg_prob = pred_seg.detach().cpu()

                    data["pred_seg"] = pred_seg

                    if has_mask:
                        data["mask"] = data["mask"].to(self.device)

                    data_decollated = [
                        self.post_transforms(item) for item in decollate_batch(data)
                    ]

                    pred_seg_batch = from_engine(["pred_seg"])(data_decollated)

                    if has_mask:
                        y_seg_batch = from_engine(["mask"])(data_decollated)

                else:
                    data_decollated = list(decollate_batch(data))

                for task_name in classification_tasks:

                    task_prediction_key = f"pred_{task_name}"
                    task_label_key = f"label_{task_name}"

                    task_predictions = from_engine([task_prediction_key])(
                        data_decollated
                    )

                    task_predictions_argmax = [
                        torch.argmax(prediction).cpu().detach().numpy().item()
                        for prediction in task_predictions
                    ]

                    task_labels_argmax = None

                    if task_label_key in data:

                        labels_available = True

                        task_labels = from_engine([task_label_key])(data_decollated)

                        task_labels_argmax = [
                            (
                                torch.argmax(label).cpu().detach().numpy().item()
                                if torch.any(label != 0).item()
                                else -1
                            )
                            for label in task_labels
                        ]

                    for index in range(len(image)):

                        probabilities[task_name].append(
                            task_predictions[index].cpu().detach().numpy().tolist()
                        )

                        predictions[task_name].append(task_predictions_argmax[index])

                        if task_labels_argmax is not None:
                            ground_truth[task_name].append(task_labels_argmax[index])

                if has_segmentation and save_predictions:

                    pred_seg_batch_save = [
                        nib.Nifti1Image(
                            torch.squeeze(pred_seg_batch[index])
                            .cpu()
                            .detach()
                            .numpy()
                            .astype(np.uint8),
                            image_affines[index],
                        )
                        for index in range(len(image))
                    ]

                    for index in range(len(image)):

                        nib.save(
                            pred_seg_batch_save[index],
                            os.path.join(
                                save_results[index],
                                constants.PREDICTION_FILE_NAME
                                + constants.DATA_NIFTI_EXTENSION,
                            ),
                        )

                    if has_mask:

                        y_seg_batch_save = [
                            nib.Nifti1Image(
                                torch.squeeze(y_seg_batch[index])
                                .cpu()
                                .detach()
                                .numpy()
                                .astype(np.uint8),
                                image_affines[index],
                            )
                            for index in range(len(image))
                        ]

                        for index in range(len(image)):

                            nib.save(
                                y_seg_batch_save[index],
                                os.path.join(
                                    save_results[index],
                                    constants.MASK_FILE_NAME
                                    + constants.DATA_NIFTI_EXTENSION,
                                ),
                            )

                if has_segmentation and save_prob_map:

                    pred_seg_prob_batch_save = [
                        self._probability_map_to_nifti(
                            pred_seg_prob[index],
                            image_affines[index],
                        )
                        for index in range(len(image))
                    ]

                    for index in range(len(image)):

                        nib.save(
                            pred_seg_prob_batch_save[index],
                            os.path.join(
                                save_results[index],
                                constants.PROBABILITY_MAP_FILE_NAME
                                + constants.DATA_NIFTI_EXTENSION,
                            ),
                        )

                cases.extend(case_ids)

        results_df = {
            "Case": cases,
        }

        for task_name in classification_tasks:

            if labels_available:
                results_df[constants.CLASSIFICATION_LABEL_COLUMNS[task_name]] = (
                    ground_truth[task_name]
                )

            results_df[constants.CLASSIFICATION_PROBABILITY_COLUMNS[task_name]] = (
                probabilities[task_name]
            )

            results_df[constants.CLASSIFICATION_PREDICTION_COLUMNS[task_name]] = (
                predictions[task_name]
            )

        if labels_available:

            print("MRI scans and classification labels available")

            output_file = os.path.join(
                self.metrics_dir,
                f"{constants.PREDICTIONS_SUMMARY_LABELED_STEM}{self.table_extension}",
            )

        else:

            print("Only MRI scans available")

            output_file = os.path.join(
                self.metrics_dir,
                f"{constants.PREDICTIONS_SUMMARY_UNLABELED_STEM}{self.table_extension}",
            )

        write_output_table(
            table=pd.DataFrame.from_dict(results_df),
            table_path=output_file,
            index=False,
        )
