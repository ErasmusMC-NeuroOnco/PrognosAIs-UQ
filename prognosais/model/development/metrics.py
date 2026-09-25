import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.metrics import DiceMetric
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.IO.output import read_output_table, write_output_table


def dice_score(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the Dice score for one pair of prediction and target tensors.

    Args:
        predictions: torch.Tensor
            Model output tensor.
        targets: torch.Tensor
            Ground-truth tensor.

    Returns:
        torch.Tensor: Dice score value.
    """

    dice_measure = DiceMetric(include_background=False, reduction="mean")
    return dice_measure(predictions, targets)


class ClassificationMetrics:
    """
    Compute and export classification and segmentation metrics.

    The class reads the labeled prediction summary produced during evaluation,
    derives ROC, PR, confusion matrices, classification reports, and
    segmentation statistics.
    """

    def __init__(self, config_file_dir: str, results_dir: str) -> None:
        """
        Initialize the metrics helper.

        Args:
            config_file_dir: Path to the configuration file.
            results_dir: Directory containing evaluation outputs.
        """
        self.config_file_dir = config_file_dir
        self.results_dir = results_dir
        self.metrics_dir = os.path.join(self.results_dir, constants.METRICS_DIR_NAME)
        self.predictions_dir = os.path.join(
            self.results_dir, constants.PREDICTIONS_DIR_NAME
        )
        self.config_file = configIO.Config(self.config_file_dir)
        self.image_extension = self.config_file.output_images_format
        self.table_extension = self.config_file.output_tables_format

        self.labeled_predictions_dir = os.path.join(
            self.metrics_dir,
            f"{constants.PREDICTIONS_SUMMARY_LABELED_STEM}{self.table_extension}",
        )
        self.segmentation_metrics_dir = os.path.join(
            self.metrics_dir,
            f"{constants.SEGMENTATION_METRICS_STEM}{self.table_extension}",
        )
        self.results_df = read_output_table(
            self.labeled_predictions_dir, index_col=None
        )

        self.missing_value = self.config_file.data_test_missing_value

        self.classification_tasks = self.config_file.classification_tasks

        self.labels = {}
        self.labels_valid = {}
        self.preds = {}
        self.preds_am = {}

        for task_name in self.classification_tasks:

            labels = np.array(
                self.results_df[
                    constants.CLASSIFICATION_LABEL_COLUMNS[task_name]
                ].values
            )

            preds = np.array(
                self.results_df[
                    constants.CLASSIFICATION_PROBABILITY_COLUMNS[task_name]
                ].values
            )
            preds = np.array([ast.literal_eval(arr) for arr in preds])

            valid = labels != self.missing_value

            self.labels[task_name] = labels
            self.labels_valid[task_name] = labels[valid]
            self.preds[task_name] = preds
            self.preds_am[task_name] = [np.argmax(pred) for pred in preds[valid]]

    def _write_segmentation_metrics(self) -> None:
        """Persist Dice and Hausdorff values in a dedicated segmentation table."""

        metric_columns = [
            column_name
            for column_name in (
                constants.SEGMENTATION_DICE_SCORE_COLUMN,
                constants.SEGMENTATION_HD_COLUMN,
            )
            if column_name in self.results_df.columns
        ]
        if not metric_columns:
            return

        write_output_table(
            table=self.results_df[[constants.CASE_COLUMN_NAME, *metric_columns]],
            table_path=self.segmentation_metrics_dir,
            index=False,
        )

    def _get_valid_samples(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Return the valid samples for each classification task.

        Returns:
            dict[str, dict[str, np.ndarray]]: Per-task labels and predictions.
        """

        valid_samples = {}

        for task_name in self.classification_tasks:

            valid = self.labels[task_name] != self.missing_value

            labels = self.labels[task_name][valid]
            preds = self.preds[task_name][valid]

            sample = {
                "labels": labels,
                "preds": preds,
            }

            if constants.NUM_CLASSES_PER_LABEL[task_name] > 2:
                sample["labels_oh"] = (
                    LabelBinarizer()
                    .fit(range(constants.NUM_CLASSES_PER_LABEL[task_name]))
                    .transform(labels)
                )

            valid_samples[task_name] = sample

        return valid_samples

    def generate_roc(self, fold_name: str | None) -> None:
        """
        Generate ROC curves for the classification tasks.

        Args:
            fold_name: Optional fold identifier used in output file names.
        """
        if fold_name is not None:
            assert isinstance(fold_name, str), "Fold name should be a string."

        valid_samples = self._get_valid_samples()
        chance_curve = np.linspace(0, 1, 100)
        auc_format = "{:.2f}"

        binary_curves = []
        multiclass_curves = {}
        all_curves = []

        for task_name in self.classification_tasks:
            labels = valid_samples[task_name]["labels"]
            preds = valid_samples[task_name]["preds"]

            if constants.NUM_CLASSES_PER_LABEL[task_name] == 2:
                fpr, tpr, _ = metrics.roc_curve(
                    y_true=labels,
                    y_score=preds[:, 1],
                )
                auc = metrics.auc(fpr, tpr)
                label = (
                    f"{constants.CLASSIFICATION_DISPLAY_NAMES[task_name]}, "
                    f"AUC={auc_format.format(auc)}, n={len(labels)}"
                )
                curve = (fpr, tpr, label)
                binary_curves.append(curve)
                all_curves.append(curve)

            else:
                task_curves = []
                labels_oh = valid_samples[task_name]["labels_oh"]

                for class_idx, class_name in enumerate(
                    constants.CLASS_NAMES_PER_LABEL[task_name]
                ):
                    fpr, tpr, _ = metrics.roc_curve(
                        y_true=labels_oh[:, class_idx],
                        y_score=preds[:, class_idx],
                    )
                    auc = metrics.auc(fpr, tpr)
                    label = (
                        f"{class_name}, "
                        f"AUC={auc_format.format(auc)}, n={len(labels)}"
                    )
                    curve = (fpr, tpr, label)
                    task_curves.append(curve)
                    all_curves.append(curve)

                multiclass_curves[task_name] = task_curves

        if binary_curves:
            plt.figure()
            for fpr, tpr, label in binary_curves:
                plt.plot(fpr, tpr, label=label)
            plt.plot(
                chance_curve,
                chance_curve,
                linestyle="--",
                color="gray",
                label=constants.ROC_CHANCE_LEVEL_LABEL,
            )
            plt.xlabel("1 - Specificity (FPR)")
            plt.ylabel("Sensitivity (TPR)")
            plt.legend(loc=0)

            binary_stem = "_".join(
                constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                    constants.FILENAME_STEM_KEY
                ]
                for task_name in self.classification_tasks
                if constants.NUM_CLASSES_PER_LABEL[task_name] == 2
            )
            filename = (
                f"ROC_{binary_stem}{self.image_extension}"
                if fold_name is None
                else f"ROC_{binary_stem}_fold{fold_name}{self.image_extension}"
            )
            plt.savefig(os.path.join(self.metrics_dir, filename))
            plt.close()

        for task_name, task_curves in multiclass_curves.items():
            plt.figure()
            for fpr, tpr, label in task_curves:
                plt.plot(fpr, tpr, label=label)
            plt.plot(
                chance_curve,
                chance_curve,
                linestyle="--",
                color="gray",
                label=constants.ROC_CHANCE_LEVEL_LABEL,
            )
            plt.xlabel("1 - Specificity (FPR)")
            plt.ylabel("Sensitivity (TPR)")
            plt.legend(loc=0)

            task_stem = constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                constants.FILENAME_STEM_KEY
            ]
            filename = (
                f"ROC_{task_stem}{self.image_extension}"
                if fold_name is None
                else f"ROC_{task_stem}_fold{fold_name}{self.image_extension}"
            )
            plt.savefig(os.path.join(self.metrics_dir, filename))
            plt.close()

        if all_curves:
            plt.figure()
            for fpr, tpr, label in all_curves:
                plt.plot(fpr, tpr, label=label)
            plt.plot(
                chance_curve,
                chance_curve,
                linestyle="--",
                color="gray",
                label=constants.ROC_CHANCE_LEVEL_LABEL,
            )
            plt.xlabel("1 - Specificity (FPR)")
            plt.ylabel("Sensitivity (TPR)")
            plt.legend(loc=0)
            filename = (
                f"ROC_All{self.image_extension}"
                if fold_name is None
                else f"ROC_All_fold{fold_name}{self.image_extension}"
            )
            plt.savefig(os.path.join(self.metrics_dir, filename))
            plt.close()

    def generate_pr(self, fold_name: str | None) -> None:
        """
        Generate precision-recall curves for the classification tasks.

        Args:
            fold_name: Optional fold identifier used in output file names.
        """
        if fold_name is not None:
            assert isinstance(fold_name, str), "Fold name should be a string."

        valid_samples = self._get_valid_samples()
        auc_format = "{:.2f}"

        binary_curves = []
        multiclass_curves = {}
        all_curves = []

        for task_name in self.classification_tasks:
            labels = valid_samples[task_name]["labels"]
            preds = valid_samples[task_name]["preds"]

            if constants.NUM_CLASSES_PER_LABEL[task_name] == 2:
                precision, recall, _ = metrics.precision_recall_curve(
                    y_true=labels,
                    y_score=preds[:, 1],
                )
                auc = metrics.auc(recall, precision)
                label = (
                    f"{constants.CLASSIFICATION_DISPLAY_NAMES[task_name]}, "
                    f"AUC={auc_format.format(auc)}, n={len(labels)}"
                )
                curve = (recall, precision, label)
                binary_curves.append(curve)
                all_curves.append(curve)

            else:
                task_curves = []
                labels_oh = valid_samples[task_name]["labels_oh"]

                for class_idx, class_name in enumerate(
                    constants.CLASS_NAMES_PER_LABEL[task_name]
                ):
                    precision, recall, _ = metrics.precision_recall_curve(
                        y_true=labels_oh[:, class_idx],
                        y_score=preds[:, class_idx],
                    )
                    auc = metrics.auc(recall, precision)
                    label = (
                        f"{class_name}, "
                        f"AUC={auc_format.format(auc)}, n={len(labels)}"
                    )
                    curve = (recall, precision, label)
                    task_curves.append(curve)
                    all_curves.append(curve)

                multiclass_curves[task_name] = task_curves

        if binary_curves:
            plt.figure()
            for recall, precision, label in binary_curves:
                plt.plot(recall, precision, label=label)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc=0)

            binary_stem = "_".join(
                constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                    constants.FILENAME_STEM_KEY
                ]
                for task_name in self.classification_tasks
                if constants.NUM_CLASSES_PER_LABEL[task_name] == 2
            )
            filename = (
                f"PR_{binary_stem}{self.image_extension}"
                if fold_name is None
                else f"PR_{binary_stem}_fold{fold_name}{self.image_extension}"
            )
            plt.savefig(os.path.join(self.metrics_dir, filename))
            plt.close()

        for task_name, task_curves in multiclass_curves.items():
            plt.figure()
            for recall, precision, label in task_curves:
                plt.plot(recall, precision, label=label)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc=0)

            task_stem = constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                constants.FILENAME_STEM_KEY
            ]
            filename = (
                f"PR_{task_stem}{self.image_extension}"
                if fold_name is None
                else f"PR_{task_stem}_fold{fold_name}{self.image_extension}"
            )
            plt.savefig(os.path.join(self.metrics_dir, filename))
            plt.close()

        if all_curves:
            plt.figure()
            for recall, precision, label in all_curves:
                plt.plot(recall, precision, label=label)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc=0)
            filename = (
                f"PR_All{self.image_extension}"
                if fold_name is None
                else f"PR_All_fold{fold_name}{self.image_extension}"
            )
            plt.savefig(os.path.join(self.metrics_dir, filename))
            plt.close()

    def generate_cf(self, fold_name: str | None) -> None:
        """
        Generate confusion matrices for the classification tasks.

        Args:
            fold_name: Optional fold identifier used in output file names.
        """
        if fold_name is not None:
            assert isinstance(fold_name, str), "Fold name should be a string."

        self.confusion_matrices = {}

        for task_name in self.classification_tasks:

            n_classes = constants.NUM_CLASSES_PER_LABEL[task_name]

            cm = metrics.confusion_matrix(
                y_true=self.labels_valid[task_name],
                y_pred=self.preds_am[task_name],
                labels=range(n_classes),
            )

            self.confusion_matrices[task_name] = cm

            disp = metrics.ConfusionMatrixDisplay(
                cm,
                display_labels=constants.CLASS_NAMES_PER_LABEL[task_name],
            )

            plt.figure()
            disp.plot(cmap=plt.cm.Blues)

            filename = (
                f"{constants.CONFUSION_MATRIX_FILE_STEMS[task_name]}"
                f"{self.image_extension}"
                if fold_name is None
                else f"{constants.CONFUSION_MATRIX_FILE_STEMS[task_name]}"
                f"_fold{fold_name}{self.image_extension}"
            )

            plt.savefig(os.path.join(self.metrics_dir, filename))

    def calculate_tp_tn_fp_fn(self) -> None:
        """
        Calculate true positives, true negatives, false positives, and false
        negatives for each task.
        """

        self.tp = {}
        self.tn = {}
        self.fp = {}
        self.fn = {}

        for task_name in self.classification_tasks:

            cm = self.confusion_matrices[task_name]

            self.tp[task_name] = {}
            self.tn[task_name] = {}
            self.fp[task_name] = {}
            self.fn[task_name] = {}

            for i in range(cm.shape[0]):

                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fp - fn

                self.tp[task_name][i] = tp
                self.fp[task_name][i] = fp
                self.fn[task_name][i] = fn
                self.tn[task_name][i] = tn

    def calculate_specificity(self) -> None:
        """
        Calculate task-specific specificity values from the confusion counts.
        """

        self.spec = {}

        for task_name in self.classification_tasks:

            self.spec[task_name] = {}

            for class_label in self.tn[task_name]:

                tn = self.tn[task_name][class_label]
                fp = self.fp[task_name][class_label]

                self.spec[task_name][class_label] = (
                    tn / (tn + fp) if (tn + fp) > 0 else 0
                )

    def generate_classification_report(self) -> None:
        """
        Generate classification reports for all tasks and save them to disk.
        """
        self.support_values = {}

        for task_name in self.classification_tasks:

            n_classes = constants.NUM_CLASSES_PER_LABEL[task_name]

            cr = metrics.classification_report(
                y_true=self.labels_valid[task_name],
                y_pred=self.preds_am[task_name],
                labels=range(n_classes),
                output_dict=True,
                zero_division=0,
            )

            cr.update(
                {
                    constants.ACCURACY_ROW_NAME: {
                        constants.PRECISION_COLUMN_NAME: None,
                        constants.RECALL_COLUMN_NAME: None,
                        constants.F1_SCORE_COLUMN_NAME: cr[constants.ACCURACY_ROW_NAME],
                        constants.SUPPORT_COLUMN_NAME: cr[constants.MACRO_AVG_ROW_NAME][
                            constants.SUPPORT_COLUMN_NAME
                        ],
                    }
                }
            )

            cr_df = pd.DataFrame(cr).transpose()

            rename_index = {
                str(i): class_name
                for i, class_name in enumerate(
                    constants.CLASSIFICATION_REPORT_INDEX_LABELS[task_name]
                )
            }

            rename_index.update(
                {
                    constants.ACCURACY_ROW_NAME: constants.ACCURACY_ROW_NAME,
                    constants.MACRO_AVG_ROW_NAME: constants.MACRO_AVG_ROW_NAME,
                    constants.WEIGHTED_AVG_ROW_NAME: constants.WEIGHTED_AVG_ROW_NAME,
                }
            )

            cr_df.rename(
                columns={
                    constants.PRECISION_COLUMN_NAME: constants.PRECISION_COLUMN_NAME,
                    constants.RECALL_COLUMN_NAME: constants.SENSITIVITY_COLUMN_NAME,
                    constants.F1_SCORE_COLUMN_NAME: constants.F1_SCORE_COLUMN_NAME,
                    constants.SUPPORT_COLUMN_NAME: constants.SUPPORT_COLUMN_NAME,
                },
                index=rename_index,
                inplace=True,
            )

            spec_values = list(self.spec[task_name].values())

            support = cr_df[constants.SUPPORT_COLUMN_NAME].tolist()[: len(spec_values)]
            total = cr_df[constants.SUPPORT_COLUMN_NAME].tolist()[len(spec_values)]

            support_fraction = [s / total for s in support]

            self.support_values[task_name] = support

            macro_avg_spec = np.mean(spec_values)
            weighted_avg_spec = np.sum(
                np.array(spec_values) * np.array(support_fraction)
            )

            spec_values += [None, macro_avg_spec, weighted_avg_spec]

            cr_df.insert(
                2,
                constants.SPECIFICITY_COLUMN_NAME,
                spec_values,
            )

            write_output_table(
                table=cr_df,
                table_path=os.path.join(
                    self.metrics_dir,
                    f"{constants.CLASSIFICATION_REPORT_FILE_STEMS[task_name]}"
                    f"{self.table_extension}",
                ),
                index=True,
            )

    def barplots(self, fold_name: str | None) -> None:
        """
        Generate bar plots for per-task metrics.

        Args:
            fold_name: Optional fold identifier used in output file names.
        """
        if fold_name is None:
            base_dir = os.path.dirname(self.results_dir)
        else:
            base_dir = os.path.dirname(os.path.dirname(self.results_dir))

        info_dir = os.path.join(
            base_dir,
            constants.INFORMATION_DIR_NAME,
        )
        os.makedirs(info_dir, exist_ok=True)

        for task_name in self.classification_tasks:

            labels = list(constants.CLASS_NAMES_PER_LABEL[task_name])

            plt.figure()
            plt.bar(labels, self.support_values[task_name])
            plt.yticks(self.support_values[task_name])
            plt.title(
                f"Test dataset distribution for "
                f"{constants.CLASSIFICATION_DISPLAY_NAMES[task_name]}"
            )
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    info_dir,
                    f"{constants.DISTRIBUTION_FILE_STEMS[task_name]}"
                    f"{self.image_extension}",
                )
            )

            plt.close()

    def get_hd(self) -> None:
        """
        Compute Hausdorff distance statistics for the segmentation outputs.
        """

        cases = self.results_df[constants.CASE_COLUMN_NAME].values
        hd = []

        for case in cases:

            mask_path = os.path.join(
                self.predictions_dir,
                case,
                constants.MASK_FILE_NAME + constants.DATA_NIFTI_EXTENSION,
            )
            pred_path = os.path.join(
                self.predictions_dir,
                case,
                constants.PREDICTION_FILE_NAME + constants.DATA_NIFTI_EXTENSION,
            )

            # Read image with SimpleITK
            mask_im = sitk.ReadImage(mask_path)
            pred_im = sitk.ReadImage(pred_path)

            # Hausdorff distance filter

            hausdorff_filter = sitk.HausdorffDistanceImageFilter()

            try:
                hausdorff_filter.Execute(mask_im, pred_im)
                hd.append(hausdorff_filter.GetHausdorffDistance())
            except RuntimeError:
                print(f"Error calculating Hausdorff Distance for case {case}")
                print(
                    f"Number of non-zero pixels in mask: {sitk.GetArrayFromImage(mask_im).sum()}"
                )
                print(
                    f"Number of non-zero pixels in prediction: {sitk.GetArrayFromImage(pred_im).sum()}"
                )
                hd.append(9999999)
            # metrics = sg.write_metrics(
            #     labels=[1], gdth_path=mask_path, pred_path=pred_path, metrics=["hd"]
            # )
            # hd.append(metrics[0]["hd"][0])

            # hd.append(hausdorff_filter.GetHausdorffDistance())

        self.results_df = self.results_df.drop(
            columns=[constants.SEGMENTATION_HD_COLUMN],
            errors="ignore",
        )
        self.results_df.insert(2, constants.SEGMENTATION_HD_COLUMN, hd)
        self._write_segmentation_metrics()

    def get_dice(self) -> None:
        """
        Compute Dice score statistics for the segmentation outputs.
        """

        cases = self.results_df[constants.CASE_COLUMN_NAME].values
        dice = []

        for case in cases:

            mask_path = os.path.join(
                self.predictions_dir,
                case,
                constants.MASK_FILE_NAME + constants.DATA_NIFTI_EXTENSION,
            )
            pred_path = os.path.join(
                self.predictions_dir,
                case,
                constants.PREDICTION_FILE_NAME + constants.DATA_NIFTI_EXTENSION,
            )

            # Read image with SimpleITK
            mask_im = sitk.ReadImage(mask_path)
            pred_im = sitk.ReadImage(pred_path)

            # Hausdorff distance filter

            overlap_filter = sitk.LabelOverlapMeasuresImageFilter()

            overlap_filter.Execute(mask_im, pred_im)

            # metrics = sg.write_metrics(
            #     labels=[1], gdth_path=mask_path, pred_path=pred_path, metrics=["hd"]
            # )
            # hd.append(metrics[0]["hd"][0])

            dice.append(overlap_filter.GetDiceCoefficient())

        self.results_df = self.results_df.drop(
            columns=[constants.SEGMENTATION_DICE_SCORE_COLUMN],
            errors="ignore",
        )
        self.results_df.insert(1, constants.SEGMENTATION_DICE_SCORE_COLUMN, dice)
        self._write_segmentation_metrics()

    def export_metrics_stats_df(self) -> None:
        """
        Export the aggregated metrics statistics using the configured table format.
        """
        """
        Create a DataFrame to store metric statistics and save it using the
        configured table format.
        """

        dice_mean = np.mean(
            self.results_df[constants.SEGMENTATION_DICE_SCORE_COLUMN].values
        )
        dice_median = np.median(
            self.results_df[constants.SEGMENTATION_DICE_SCORE_COLUMN].values
        )
        dice_max = np.max(
            self.results_df[constants.SEGMENTATION_DICE_SCORE_COLUMN].values
        )
        dice_min = np.min(
            self.results_df[constants.SEGMENTATION_DICE_SCORE_COLUMN].values
        )

        hd_mean = np.mean(self.results_df[constants.SEGMENTATION_HD_COLUMN].values)
        hd_median = np.median(self.results_df[constants.SEGMENTATION_HD_COLUMN].values)
        hd_max = np.max(self.results_df[constants.SEGMENTATION_HD_COLUMN].values)
        hd_min = np.min(self.results_df[constants.SEGMENTATION_HD_COLUMN].values)

        stats_df = {
            constants.METRIC_COLUMN_NAME: [
                constants.SEGMENTATION_DICE_SCORE_COLUMN,
                constants.HAUSDORFF_DISTANCE_METRIC_NAME,
            ],
            "Min": [dice_min, hd_min],
            "Max": [dice_max, hd_max],
            "Mean": [dice_mean, hd_mean],
            "Median": [dice_median, hd_median],
        }
        self.stats_df = pd.DataFrame(stats_df)
        write_output_table(
            table=self.stats_df,
            table_path=os.path.join(
                self.metrics_dir,
                f"{constants.METRICS_STATISTICS_STEM}{self.table_extension}",
            ),
            index=False,
        )

    def get_bp_dice_hd(self) -> None:
        """
        Recompute segmentation metrics and generate their boxplots.

        Dice score and Hausdorff distance are always calculated from the current
        prediction and ground-truth files. Existing table values are never used as
        a cache because they may belong to an earlier inference run.
        """

        self.get_dice()
        self.get_hd()

        self.export_metrics_stats_df()

        dice_score = np.array(
            self.results_df[constants.SEGMENTATION_DICE_SCORE_COLUMN].values
        )
        hd = np.array(self.results_df[constants.SEGMENTATION_HD_COLUMN].values)

        mean_dice = np.mean(dice_score)
        median_dice = np.median(dice_score)
        mean_hd = np.mean(hd)
        median_hd = np.median(hd)
        max_hd = np.max(hd)

        # Dice score boxplot
        plt.figure()
        bp_dice = plt.boxplot(dice_score, showmeans=True, meanline=True)

        if median_dice > mean_dice:
            plt.text(
                1,
                mean_dice - 0.03,
                f"{mean_dice:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )
            plt.text(
                1,
                median_dice + 0.01,
                f"{median_dice:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )
        else:
            plt.text(
                1,
                mean_dice + 0.01,
                f"{mean_dice:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )
            plt.text(
                1,
                median_dice - 0.03,
                f"{median_dice:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )

        plt.ylabel(constants.SEGMENTATION_DICE_SCORE_COLUMN)
        plt.xticks([])  # Remove x-axis labels
        plt.legend([bp_dice["medians"][0], bp_dice["means"][0]], ["median", "mean"])
        plt.savefig(
            os.path.join(
                self.metrics_dir,
                f"{constants.BOXPLOT_FILE_STEMS['dice']}{self.image_extension}",
            )
        )

        # Hausdorff distance boxplot
        plt.figure()
        bp_hd = plt.boxplot(hd, showmeans=True, meanline=True)

        if median_hd > mean_hd:
            plt.text(
                1,
                mean_hd - 0.03 * max_hd,
                f"{mean_hd:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )
            plt.text(
                1,
                median_hd + 0.01 * max_hd,
                f"{median_hd:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )
        else:
            plt.text(
                1,
                mean_hd + 0.01 * max_hd,
                f"{mean_hd:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )
            plt.text(
                1,
                median_hd - 0.03 * max_hd,
                f"{median_hd:.2f}",
                horizontalalignment="center",
                fontsize=8,
            )

        plt.title("Boxplot for the Hausdorff distance")
        plt.ylabel(f"{constants.HAUSDORFF_DISTANCE_METRIC_NAME} (mm)")
        plt.xticks([])  # Remove x-axis labels
        plt.legend([bp_hd["medians"][0], bp_hd["means"][0]], ["median", "mean"])
        plt.savefig(
            os.path.join(
                self.metrics_dir,
                f"{constants.BOXPLOT_FILE_STEMS['hd']}{self.image_extension}",
            )
        )
