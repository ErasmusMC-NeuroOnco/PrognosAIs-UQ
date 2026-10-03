"""Monte Carlo dropout inference for configurable PrognosAIs models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from prognosais.IO import constants
from prognosais.IO.output import write_output_table
from prognosais.UQ.metrics import ExpectedEntropy, PredictiveEntropy
from prognosais.UQ.task_utils import (
    ClassificationTaskSpec,
    classification_uq_filename,
)


class MCDropout:
    """Run stochastic inference and export task-aware uncertainty outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        data_loader,
        dropout_rate: float,
        mc_samples: int,
        mcd_dir: str | Path,
        temp_dir: str | Path,
        task_specs: Iterable[ClassificationTaskSpec],
        does_segmentation: bool,
        inference_mode: str,
        labels_available: bool,
        mask_available: bool,
        image_extension: str,
        table_extension: str,
    ) -> None:
        """Initialize a Monte Carlo dropout run.

        Args:
            model: Configured CSNet model with loaded weights.
            device: Device used for stochastic forward passes.
            data_loader: Deterministically ordered inference data loader.
            dropout_rate: Dropout probability represented by this run.
            mc_samples: Number of stochastic forward passes.
            mcd_dir: Directory where final tables and figures are written.
            temp_dir: Run-local directory for full segmentation maps and case
                alignment metadata.
            task_specs: Explicit metadata for every enabled classification task.
            does_segmentation: Whether the model emits segmentation logits.
            inference_mode: Configured test-data mode, labeled or unlabeled.
            labels_available: Whether classification labels are present.
            mask_available: Whether segmentation ground truth is present.
            image_extension: File extension used for generated figures.
            table_extension: File extension used for generated tables.

        Raises:
            ValueError: If run parameters or model task heads are inconsistent.
        """

        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.dropout_rate = float(dropout_rate)
        self.dropout_rate_str = str(dropout_rate).replace(".", "")
        self.mc_samples = int(mc_samples)
        self.mcd_dir = Path(mcd_dir)
        self.temp_dir = Path(temp_dir)
        self.task_specs = list(task_specs)
        self.does_segmentation = bool(does_segmentation)
        self.inference_mode = inference_mode
        self.labels_available = bool(labels_available)
        self.mask_available = bool(mask_available)
        self.image_extension = image_extension
        self.table_extension = table_extension

        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0 <= rate < 1.")
        if self.mc_samples <= 0:
            raise ValueError("mc_samples must be a positive integer.")
        if not self.task_specs:
            raise ValueError("At least one classification task is required.")
        if self.inference_mode not in constants.SUPPORTED_INFERENCE_MODES:
            raise ValueError(
                f"Unsupported inference mode '{self.inference_mode}'. Supported "
                "values: " + ", ".join(constants.SUPPORTED_INFERENCE_MODES)
            )
        expected_labels_available = (
            self.inference_mode == constants.INFERENCE_MODE_LABELED
        )
        if self.labels_available != expected_labels_available:
            raise ValueError(
                "MCD labels_available does not match inference_mode "
                f"'{self.inference_mode}'."
            )
        if self.mask_available and not self.labels_available:
            raise ValueError(
                "An unlabeled MCD run cannot include segmentation ground truth."
            )
        if self.mask_available and not self.does_segmentation:
            raise ValueError(
                "A segmentation mask cannot be used by a classification-only model."
            )
        if self.image_extension not in constants.SUPPORTED_IMAGE_OUTPUT_FORMATS:
            supported_formats = ", ".join(constants.SUPPORTED_IMAGE_OUTPUT_FORMATS)
            raise ValueError(
                "Unsupported image_extension "
                f"'{self.image_extension}'. Choose one of: "
                f"{supported_formats}."
            )
        if self.table_extension not in constants.SUPPORTED_TABLE_OUTPUT_FORMATS:
            supported_formats = ", ".join(constants.SUPPORTED_TABLE_OUTPUT_FORMATS)
            raise ValueError(
                "Unsupported table_extension "
                f"'{self.table_extension}'. Choose one of: "
                f"{supported_formats}."
            )

        expected_tasks = {task.key for task in self.task_specs}
        model_tasks = set(getattr(self.model, "classification_tasks", {}))
        if model_tasks != expected_tasks:
            raise ValueError(
                "Configured UQ tasks do not match the model heads. "
                f"UQ tasks: {sorted(expected_tasks)}; model tasks: "
                f"{sorted(model_tasks)}. Use the frozen configuration that "
                "belongs to the checkpoint."
            )

        self.mcd_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_paths: dict[str, Path] = {}
        self.UQ_data_by_task: dict[str, pd.DataFrame] = {}
        self.n_cases: int | None = None
        self.dropout_modules_enabled: int | None = None

        print(f"Temporary directory is {self.temp_dir}")
        print(f"Device is {self.device}")

    def _enable_dropout(self) -> int:
        """Enable all dropout variants while leaving other layers in eval mode.

        Returns:
            Number of dropout modules placed in training mode.
        """

        enabled = 0
        for module in self.model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()
                enabled += 1
        return enabled

    @staticmethod
    def _extract_case_ids(data: dict, batch_size: int) -> list[str]:
        """Extract case identifiers from an inference batch.

        Case identifiers are required and must be emitted by DataGenerator.

        Args:
            data: Inference batch dictionary.
            batch_size: Number of images in the batch.

        Returns:
            Case identifiers in batch order.

        Raises:
            ValueError: If the required case_id field is missing or its count
                differs from the image batch size.
        """

        if constants.CASE_ID_DATA_KEY not in data:
            raise ValueError(
                "Inference batch does not contain the required "
                f"'{constants.CASE_ID_DATA_KEY}' field. Build MCD input data "
                "with DataGenerator."
            )
        raw_case_ids = data[constants.CASE_ID_DATA_KEY]
        if isinstance(raw_case_ids, str):
            case_ids = [raw_case_ids]
        else:
            case_ids = [str(case_id) for case_id in raw_case_ids]

        if len(case_ids) != batch_size:
            raise ValueError(
                f"Recovered {len(case_ids)} case IDs for a batch of "
                f"{batch_size} images."
            )
        return case_ids

    def _validate_model_outputs(
        self,
        segmentation_logits: torch.Tensor | None,
        classification_logits: dict[str, torch.Tensor],
    ) -> None:
        """Validate model output keys and segmentation capability.

        Args:
            segmentation_logits: Optional segmentation logits from the model.
            classification_logits: Classification logits keyed by task.

        Raises:
            ValueError: If outputs disagree with the configured UQ tasks or
                segmentation mode.
        """

        expected_tasks = {task.key for task in self.task_specs}
        output_tasks = set(classification_logits)
        if output_tasks != expected_tasks:
            raise ValueError(
                "Model classification outputs do not match configured tasks. "
                f"Expected {sorted(expected_tasks)}, got {sorted(output_tasks)}."
            )
        if self.does_segmentation and segmentation_logits is None:
            raise ValueError(
                "The configuration requires segmentation, but the model returned None."
            )
        if not self.does_segmentation and segmentation_logits is not None:
            raise ValueError(
                "The configuration is classification-only, but the model returned "
                "segmentation logits."
            )

    @staticmethod
    def _dice_scores(
        true_one_hot: np.ndarray,
        predicted_probabilities: np.ndarray,
    ) -> np.ndarray:
        """Compute binary tumor Dice scores with SimpleITK for one batch.

        Args:
            true_one_hot: Ground-truth one-hot masks shaped as cases by classes
                by x by y by z.
            predicted_probabilities: Predicted probabilities with the same
                dimensional convention.

        Returns:
            One Dice score per case. Two empty masks receive a score of one.
        """

        true_mask = np.argmax(true_one_hot, axis=1).astype(bool)
        predicted_mask = np.argmax(predicted_probabilities, axis=1).astype(bool)
        dice = np.empty(len(true_mask), dtype=float)

        for case_index, (true_case, predicted_case) in enumerate(
            zip(true_mask, predicted_mask)
        ):
            if not np.any(true_case) and not np.any(predicted_case):
                dice[case_index] = 1.0
                continue

            true_image = sitk.GetImageFromArray(true_case.astype(np.uint8))
            predicted_image = sitk.GetImageFromArray(
                predicted_case.astype(np.uint8)
            )
            overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_filter.Execute(true_image, predicted_image)
            dice[case_index] = overlap_filter.GetDiceCoefficient()

        return dice

    def _write_case_metadata(
        self,
        cases_id: list[str],
        cases_by_task: dict[str, list[str]],
        labels_by_task: dict[str, np.ndarray],
        labels_seg: np.ndarray | None,
    ) -> Path:
        """Write dynamic case and label metadata for downstream aggregation.

        Args:
            cases_id: All cases in segmentation and inference order.
            cases_by_task: Cases with known labels for each task, or all cases
                for an unlabeled run.
            labels_by_task: Known one-hot labels keyed by task.
            labels_seg: Optional segmentation ground-truth tensor.

        Returns:
            Path to the written metadata pickle.
        """

        metadata: dict[str, object] = {
            "schema_version": constants.UQ_SCHEMA_VERSION,
            "cases_id": cases_id,
            "task_keys": [task.key for task in self.task_specs],
            "cases_by_task": cases_by_task,
            "labels_by_task": labels_by_task,
        }
        for task in self.task_specs:
            metadata[f"cases_{task.key}"] = cases_by_task[task.key]
            if task.key in labels_by_task:
                metadata[f"labels_{task.key}"] = labels_by_task[task.key]
        if labels_seg is not None:
            metadata["labels_seg"] = labels_seg

        output_path = self.temp_dir / constants.SEGMENTATION_LABELS_FILENAME
        with output_path.open("wb") as output_file:
            pickle.dump(metadata, output_file)
        return output_path

    def run(self) -> dict[str, Path]:
        """Run stochastic inference and write all configured UQ outputs.

        Returns:
            Mapping of descriptive artifact names to written paths.

        Raises:
            ValueError: If the data loader is empty, case order changes between
                passes, labels are malformed, or output dimensions disagree.
        """

        if len(self.data_loader.dataset) == 0:
            raise ValueError("Cannot run MCD on an empty dataset.")

        predictive_entropy = {
            task.key: PredictiveEntropy(
                class_axis=-1,
                epsilon=constants.ENTROPY_EPSILON,
            )
            for task in self.task_specs
        }
        expected_entropy = {
            task.key: ExpectedEntropy(
                class_axis=-1,
                epsilon=constants.ENTROPY_EPSILON,
            )
            for task in self.task_specs
        }
        mean_probabilities: dict[str, np.ndarray | None] = {
            task.key: None for task in self.task_specs
        }
        segmentation_pe = (
            PredictiveEntropy(
                class_axis=1,
                epsilon=constants.ENTROPY_EPSILON,
            )
            if self.does_segmentation
            else None
        )
        segmentation_ee = (
            ExpectedEntropy(
                class_axis=1,
                epsilon=constants.ENTROPY_EPSILON,
            )
            if self.does_segmentation
            else None
        )
        mean_segmentation: np.ndarray | None = None

        cases_id: list[str] = []
        cases_by_task = {task.key: [] for task in self.task_specs}
        label_chunks = {task.key: [] for task in self.task_specs}
        segmentation_label_chunks: list[np.ndarray] = []
        dice_score_report: dict[str, list[float] | list[str]] = {}
        enabled_dropout_modules: int | None = None

        for sample_index in range(self.mc_samples):
            print(f"Running Monte Carlo sample {sample_index}", flush=True)
            self.model.eval()
            enabled_dropout_modules = self._enable_dropout()
            if (
                sample_index == 0
                and self.dropout_rate > 0
                and enabled_dropout_modules == 0
            ):
                raise ValueError(
                    "MCD requested a non-zero dropout rate, but the configured "
                    "model contains no dropout modules."
                )

            sample_cases: list[str] = []
            task_probability_chunks = {task.key: [] for task in self.task_specs}
            segmentation_probability_chunks: list[np.ndarray] = []
            sample_dice_chunks: list[np.ndarray] = []

            for data in self.data_loader:
                image = data["image"].to(self.device)
                batch_cases = self._extract_case_ids(data, batch_size=len(image))
                sample_cases.extend(batch_cases)

                with torch.inference_mode():
                    segmentation_logits, classification_logits = self.model(image)
                self._validate_model_outputs(
                    segmentation_logits=segmentation_logits,
                    classification_logits=classification_logits,
                )

                for task in self.task_specs:
                    probabilities = (
                        F.softmax(classification_logits[task.key], dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if probabilities.shape[1] != task.n_classes:
                        raise ValueError(
                            f"Task '{task.key}' returned {probabilities.shape[1]} "
                            f"classes; expected {task.n_classes}."
                        )

                    if self.labels_available:
                        label_key = f"label_{task.key}"
                        if label_key not in data:
                            raise ValueError(
                                f"Labeled run is missing batch key '{label_key}'."
                            )
                        labels = data[label_key].detach().cpu().numpy()
                        if labels.shape != probabilities.shape:
                            raise ValueError(
                                f"Task '{task.key}' labels have shape {labels.shape}; "
                                f"probabilities have shape {probabilities.shape}."
                            )
                        known = labels.sum(axis=1) > 0
                        probabilities = probabilities[known]
                        if sample_index == 0:
                            label_chunks[task.key].append(labels[known])
                            cases_by_task[task.key].extend(
                                np.asarray(batch_cases)[known].tolist()
                            )
                    elif sample_index == 0:
                        cases_by_task[task.key].extend(batch_cases)

                    task_probability_chunks[task.key].append(probabilities)

                if self.does_segmentation:
                    segmentation_probabilities = (
                        F.softmax(segmentation_logits, dim=1).detach().cpu().numpy()
                    )
                    if (
                        segmentation_probabilities.shape[1]
                        != constants.NUM_SEGMENTATION_CLASSES
                    ):
                        raise ValueError(
                            "Segmentation output has "
                            f"{segmentation_probabilities.shape[1]} classes; expected "
                            f"{constants.NUM_SEGMENTATION_CLASSES}."
                        )
                    segmentation_probability_chunks.append(segmentation_probabilities)

                    if self.mask_available:
                        if "mask" not in data:
                            raise ValueError(
                                "Segmentation ground truth was requested, but the "
                                "batch has no mask key."
                            )
                        labels_seg_batch = data["mask"].detach().cpu().numpy()
                        if sample_index == 0:
                            segmentation_label_chunks.append(labels_seg_batch)
                        sample_dice_chunks.append(
                            self._dice_scores(
                                true_one_hot=labels_seg_batch,
                                predicted_probabilities=segmentation_probabilities,
                            )
                        )

            if len(sample_cases) != len(set(sample_cases)):
                raise ValueError("The MCD data loader produced duplicated case IDs.")
            if sample_index == 0:
                cases_id = sample_cases
            elif sample_cases != cases_id:
                raise ValueError(
                    "Case order changed between MCD passes. Use shuffle=False and "
                    "a deterministic sampler."
                )

            for task in self.task_specs:
                task_probabilities = np.concatenate(
                    task_probability_chunks[task.key], axis=0
                )
                if len(task_probabilities) == 0:
                    raise ValueError(
                        f"No cases are available for classification task '{task.key}'."
                    )
                if mean_probabilities[task.key] is None:
                    mean_probabilities[task.key] = np.zeros_like(
                        task_probabilities, dtype=np.float64
                    )
                elif mean_probabilities[task.key].shape != task_probabilities.shape:
                    raise ValueError(
                        f"Task '{task.key}' output shape changed between MCD passes."
                    )
                mean_probabilities[task.key] += task_probabilities
                predictive_entropy[task.key].update(task_probabilities)
                expected_entropy[task.key].update(task_probabilities)

            if self.does_segmentation:
                sample_segmentation = np.concatenate(
                    segmentation_probability_chunks, axis=0
                )
                if mean_segmentation is None:
                    mean_segmentation = np.zeros_like(
                        sample_segmentation, dtype=np.float64
                    )
                elif mean_segmentation.shape != sample_segmentation.shape:
                    raise ValueError(
                        "Segmentation output shape changed between MCD passes."
                    )
                mean_segmentation += sample_segmentation
                segmentation_pe.update(sample_segmentation)
                segmentation_ee.update(sample_segmentation)

                if self.mask_available:
                    dice_score_report[f"dice_sample_{sample_index}"] = np.concatenate(
                        sample_dice_chunks
                    ).tolist()

        labels_by_task = {}
        if self.labels_available:
            labels_by_task = {
                task.key: np.concatenate(label_chunks[task.key], axis=0)
                for task in self.task_specs
            }
        labels_seg = (
            np.concatenate(segmentation_label_chunks, axis=0)
            if self.mask_available
            else None
        )
        metadata_path = self._write_case_metadata(
            cases_id=cases_id,
            cases_by_task=cases_by_task,
            labels_by_task=labels_by_task,
            labels_seg=labels_seg,
        )
        self.output_paths["case_metadata"] = metadata_path

        run_suffix = f"do{self.dropout_rate_str}_{self.mc_samples}s"
        for task in self.task_specs:
            task_mean = mean_probabilities[task.key] / float(self.mc_samples)
            task_pe = predictive_entropy[task.key].compute()
            task_ee = expected_entropy[task.key].compute()
            task_mi = task_pe - task_ee
            predicted_class = np.argmax(task_mean, axis=1)

            table_data: dict[str, object] = {
                constants.CASE_COLUMN_NAME: cases_by_task[task.key],
                constants.TASK_COLUMN_NAME: task.display_name,
            }
            for class_idx in range(task.n_classes):
                table_data[
                    constants.CONFIDENCE_LABEL_COLUMN_TEMPLATE.format(
                        class_idx=class_idx
                    )
                ] = task_mean[:, class_idx]

            if self.labels_available:
                true_class = np.argmax(labels_by_task[task.key], axis=1)
                error = (predicted_class != true_class).astype(int)
                table_data[constants.TRUE_CLASS_COLUMN_NAME] = true_class
            table_data[constants.PREDICTED_CLASS_COLUMN_NAME] = predicted_class
            table_data[constants.PREDICTIVE_ENTROPY_COLUMN_NAME] = task_pe
            table_data[constants.EXPECTED_ENTROPY_COLUMN_NAME] = task_ee
            table_data[constants.MUTUAL_INFORMATION_COLUMN_NAME] = task_mi
            if self.labels_available:
                table_data[constants.CORRECTLY_CLASSIFIED_COLUMN_NAME] = np.where(
                    error == 0, "yes", "no"
                )
                table_data[constants.ERROR_COLUMN_NAME] = error

            task_table = pd.DataFrame.from_dict(table_data)
            self.UQ_data_by_task[task.key] = task_table
            setattr(self, f"UQ_data_{task.key}", task_table)

            canonical_path = self.mcd_dir / classification_uq_filename(
                task,
                run_suffix,
                self.table_extension,
            )
            write_output_table(task_table, canonical_path, index=False)
            self.output_paths[f"{task.key}_csv"] = canonical_path

        if self.does_segmentation:
            mean_segmentation /= float(self.mc_samples)
            segmentation_predictive_entropy = segmentation_pe.compute()
            segmentation_expected_entropy = segmentation_ee.compute()
            segmentation_mutual_information = (
                segmentation_predictive_entropy - segmentation_expected_entropy
            )
            predicted_segmentation = np.argmax(mean_segmentation, axis=1).astype(
                np.uint8
            )

            segmentation_table_data: dict[str, object] = {
                constants.CASE_COLUMN_NAME: cases_id,
                constants.PREDICTIVE_ENTROPY_COLUMN_NAME: np.mean(
                    segmentation_predictive_entropy, axis=(1, 2, 3)
                ),
                constants.EXPECTED_ENTROPY_COLUMN_NAME: np.mean(
                    segmentation_expected_entropy, axis=(1, 2, 3)
                ),
                constants.MUTUAL_INFORMATION_COLUMN_NAME: np.mean(
                    segmentation_mutual_information, axis=(1, 2, 3)
                ),
            }
            segmentation_maps: dict[str, object] = {
                constants.CASE_COLUMN_NAME: cases_id,
                constants.PREDICTED_CLASS_COLUMN_NAME: predicted_segmentation,
                constants.PREDICTIVE_ENTROPY_COLUMN_NAME: (
                    segmentation_predictive_entropy
                ),
                constants.EXPECTED_ENTROPY_COLUMN_NAME: (segmentation_expected_entropy),
                constants.MUTUAL_INFORMATION_COLUMN_NAME: (
                    segmentation_mutual_information
                ),
                "Mean predictions": mean_segmentation,
            }
            if self.mask_available:
                true_segmentation = np.argmax(labels_seg, axis=1).astype(np.uint8)
                ensemble_dice = self._dice_scores(
                    true_one_hot=labels_seg,
                    predicted_probabilities=mean_segmentation,
                )
                segmentation_table_data[constants.SEGMENTATION_DICE_SCORE_COLUMN] = (
                    ensemble_dice
                )
                segmentation_maps[constants.TRUE_CLASS_COLUMN_NAME] = true_segmentation
                dice_score_report["dice_mcd_ensemble"] = ensemble_dice.tolist()

            self.UQ_data_seg = pd.DataFrame.from_dict(segmentation_table_data)
            self.UQ_data_seg_extended = segmentation_maps
            segmentation_csv_path = self.mcd_dir / (
                f"UQ_seg_{run_suffix}{self.table_extension}"
            )
            segmentation_pickle_path = self.temp_dir / f"UQ_seg_{run_suffix}.pkl.gz"
            write_output_table(self.UQ_data_seg, segmentation_csv_path, index=False)
            joblib.dump(
                self.UQ_data_seg_extended,
                segmentation_pickle_path,
                compress=True,
            )
            self.output_paths["seg_csv"] = segmentation_csv_path
            self.output_paths["seg_pkl"] = segmentation_pickle_path

            if self.mask_available:
                dice_score_report = {
                    "Case ID": cases_id,
                    **dice_score_report,
                }
                dice_table = pd.DataFrame.from_dict(dice_score_report)
                mean_values = dice_table.mean(numeric_only=True).to_dict()
                dice_table = pd.concat(
                    [
                        dice_table,
                        pd.DataFrame([{"Case ID": "mean", **mean_values}]),
                    ],
                    ignore_index=True,
                )
                dice_path = self.temp_dir / (
                    f"UQ_dice_report_{run_suffix}{self.table_extension}"
                )
                write_output_table(dice_table, dice_path, index=False)
                self.output_paths["dice_csv"] = dice_path

        self.n_cases = len(cases_id)
        self.dropout_modules_enabled = enabled_dropout_modules
        self.write_manifest()
        return dict(self.output_paths)

    def write_manifest(self) -> Path:
        """Write a manifest describing inputs, capabilities, and artifacts.

        Returns:
            Path to the written JSON manifest.

        Raises:
            ValueError: If stochastic inference has not completed.
        """

        if self.n_cases is None or self.dropout_modules_enabled is None:
            raise ValueError("Run MCDropout.run before writing its manifest.")

        run_suffix = f"do{self.dropout_rate_str}_{self.mc_samples}s"
        manifest_path = self.mcd_dir / f"UQ_manifest_{run_suffix}.json"
        self.output_paths["manifest"] = manifest_path
        manifest = {
            "schema_version": constants.UQ_SCHEMA_VERSION,
            "method": constants.UQ_METHOD_MCD,
            "dropout_rate": self.dropout_rate,
            "mc_samples": self.mc_samples,
            "task_keys": [task.key for task in self.task_specs],
            "num_classes": {task.key: task.n_classes for task in self.task_specs},
            "does_segmentation": self.does_segmentation,
            "inference_mode": self.inference_mode,
            "labels_available": self.labels_available,
            "mask_available": self.mask_available,
            "n_cases": self.n_cases,
            "dropout_modules_enabled": self.dropout_modules_enabled,
            "output_format": {
                "images": self.image_extension,
                "tables": self.table_extension,
            },
            "output_paths": {
                name: str(path)
                for name, path in self.output_paths.items()
                if name != "manifest"
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def relocate_temp_outputs(self, relocated_temp_dir: str | Path) -> dict[str, Path]:
        """Update artifact paths after the run-local directory is moved.

        Args:
            relocated_temp_dir: Final location of the previously temporary
                run directory.

        Returns:
            Updated artifact mapping, including a rewritten manifest.

        Raises:
            FileNotFoundError: If relocated_temp_dir does not exist.
        """

        destination = Path(relocated_temp_dir)
        if not destination.is_dir():
            raise FileNotFoundError(
                f"Relocated MCD temporary directory does not exist: {destination}"
            )

        source = self.temp_dir
        for name, path in tuple(self.output_paths.items()):
            try:
                relative_path = Path(path).relative_to(source)
            except ValueError:
                continue
            self.output_paths[name] = destination / relative_path

        self.temp_dir = destination
        self.write_manifest()
        return dict(self.output_paths)

    def violin_plot(self) -> dict[str, Path]:
        """Generate predictive-entropy violin plots for labeled tasks.

        Returns:
            Paths to the requested-format figures keyed by task and format.

        Raises:
            ValueError: If the run did not contain classification labels or UQ
                tables have not yet been generated.
        """

        if not self.labels_available:
            raise ValueError("Violin plots require classification labels.")
        if not self.UQ_data_by_task:
            raise ValueError("Run MCDropout.run before requesting violin plots.")

        figure_paths: dict[str, Path] = {}
        run_suffix = f"do{self.dropout_rate_str}_{self.mc_samples}s"
        for task in self.task_specs:
            task_table = self.UQ_data_by_task[task.key]
            figure, axis = plt.subplots(figsize=(6.4, 4.8))
            sns.violinplot(
                data=task_table,
                x=constants.CORRECTLY_CLASSIFIED_COLUMN_NAME,
                y=constants.PREDICTIVE_ENTROPY_COLUMN_NAME,
                order=["yes", "no"],
                cut=0,
                ax=axis,
            )
            axis.set_title(f"Predictive uncertainty for {task.display_name}")
            figure.tight_layout()
            figure_path = self.mcd_dir / (
                f"vp_{task.output_name}_{run_suffix}{self.image_extension}"
            )
            save_options = {"dpi": 300} if self.image_extension == ".png" else {}
            figure.savefig(figure_path, bbox_inches="tight", **save_options)
            figure_paths[f"{task.key}_{self.image_extension.removeprefix('.')}"] = (
                figure_path
            )
            plt.close(figure)

        return figure_paths
