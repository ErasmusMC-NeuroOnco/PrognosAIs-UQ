import argparse
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

import prognosais.IO.config as configIO
import prognosais.IO.constants as constants
from prognosais.IO.output import read_output_table, write_output_table

parser = argparse.ArgumentParser(description="Compute metrics for a kfold experiment")
parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Name of the configuration file",
    metavar="configuration file",
    dest="config",
    type=str,
)

args = parser.parse_args()
this_script_dir = Path(__file__).parent.resolve()
config_dir = this_script_dir.parent.parent.resolve().joinpath("configs", args.config)
print(
    f"This job and script {os.path.abspath(__file__)} was run with the following config file: {config_dir}",
    flush=True,
)
config = configIO.Config(config_dir)


class ClassificationMetricsKfold:
    """
    Aggregate classification and segmentation metrics across folds.
    """

    def __init__(self, config_file_dir: str, results_dir: str) -> None:
        """
        Initialize the k-fold metrics helper.

        Args:
            config_file_dir: Path to the configuration file.
            results_dir: Directory containing all fold results.
        """

        self.config_file_dir = config_file_dir
        self.results_dir = results_dir
        self.config_file = configIO.Config(self.config_file_dir)
        self.image_extension = self.config_file.output_images_format
        self.table_extension = self.config_file.output_tables_format
        self.num_folds = self.config_file.test_num_folds
        self.missing_value = self.config_file.data_test_missing_value
        self.summary_dir = os.path.join(
            self.results_dir, constants.FOLDS_SUMMARY_DIR_NAME
        )
        os.makedirs(self.summary_dir, exist_ok=True)

    def get_metrics_avg(self) -> None:
        """
        Compute and export average segmentation statistics across folds.
        """

        # Initialize lists to store statistics for each metric
        dice_data = []
        hausdorff_data = []

        # Loop through each fold directory
        for fold in range(self.num_folds):
            fold_path = os.path.join(self.results_dir, f"fold_{fold}")
            metrics_file = os.path.join(
                fold_path,
                constants.RESULTS_DIR_NAME,
                constants.METRICS_DIR_NAME,
                f"{constants.METRICS_STATISTICS_STEM}{self.table_extension}",
            )

            # Ensure metrics file exists
            if os.path.isfile(metrics_file):
                # Read the fold's metrics statistics table.
                df = read_output_table(metrics_file, index_col=None)

                # Extract statistics for Dice score
                dice_row = df[
                    df[constants.METRIC_COLUMN_NAME]
                    == constants.SEGMENTATION_DICE_SCORE_COLUMN
                ].iloc[0]
                dice_data.append(
                    {
                        "Fold": f"fold_{fold}",
                        "Min": dice_row["Min"],
                        "Max": dice_row["Max"],
                        "Mean": dice_row["Mean"],
                        "Median": dice_row["Median"],
                    }
                )

                # Extract statistics for Hausdorff distance
                hausdorff_row = df[
                    df[constants.METRIC_COLUMN_NAME].str.contains(
                        constants.HAUSDORFF_DISTANCE_METRIC_NAME
                    )
                ].iloc[0]
                hausdorff_data.append(
                    {
                        "Fold": f"fold_{fold}",
                        "Min": hausdorff_row["Min"],
                        "Max": hausdorff_row["Max"],
                        "Mean": hausdorff_row["Mean"],
                        "Median": hausdorff_row["Median"],
                    }
                )

        # Convert lists to DataFrames
        dice_df = pd.DataFrame(dice_data)
        hausdorff_df = pd.DataFrame(hausdorff_data)

        # Add a row for the mean of each column
        dice_mean = dice_df.mean(numeric_only=True)
        hausdorff_mean = hausdorff_df.mean(numeric_only=True)

        dice_df.loc["Average"] = ["Average"] + dice_mean.tolist()
        hausdorff_df.loc["Average"] = ["Average"] + hausdorff_mean.tolist()

        # Save summaries using the configured table format.
        write_output_table(
            table=dice_df,
            table_path=os.path.join(
                self.summary_dir,
                f"{constants.DICE_SCORE_SUMMARY_STEM}{self.table_extension}",
            ),
            index=False,
        )
        write_output_table(
            table=hausdorff_df,
            table_path=os.path.join(
                self.summary_dir,
                f"{constants.HAUSDORFF_DISTANCE_SUMMARY_STEM}"
                f"{self.table_extension}",
            ),
            index=False,
        )

    def get_avg_cr(self) -> None:
        """
        Compute and export the average classification reports across folds.
        """

        for task_name in self.config_file.classification_tasks:

            cr = (
                f"{constants.CLASSIFICATION_REPORT_FILE_STEMS[task_name]}"
                f"{self.table_extension}"
            )

            precision = []
            sensitivity = []
            specificity = []
            f1_score = []
            accuracy = []

            for fold in range(self.num_folds):

                fold_path = os.path.join(self.results_dir, f"fold_{fold}")

                cr_df = read_output_table(
                    table_path=os.path.join(
                        fold_path,
                        constants.RESULTS_DIR_NAME,
                        constants.METRICS_DIR_NAME,
                        cr,
                    ),
                    index_col=0,
                )
                n_classes = len(cr_df) - 3
                support = list(cr_df[constants.SUPPORT_COLUMN_NAME])
                df_index = list(cr_df.index)
                precision.append(
                    np.array(cr_df[constants.PRECISION_COLUMN_NAME].iloc[:n_classes])
                )
                sensitivity.append(
                    np.array(cr_df[constants.SENSITIVITY_COLUMN_NAME].iloc[:n_classes])
                )
                specificity.append(
                    np.array(cr_df[constants.SPECIFICITY_COLUMN_NAME].iloc[:n_classes])
                )
                f1_score.append(
                    np.array(cr_df[constants.F1_SCORE_COLUMN_NAME].iloc[:n_classes])
                )
                accuracy.append(
                    cr_df.loc[constants.ACCURACY_ROW_NAME][
                        constants.F1_SCORE_COLUMN_NAME
                    ]
                )

            precision = np.column_stack(precision)
            sensitivity = np.column_stack(sensitivity)
            specificity = np.column_stack(specificity)
            f1_score = np.column_stack(f1_score)

            precision_mean = np.mean(precision, axis=1)
            sensitivity_mean = np.mean(sensitivity, axis=1)
            specificity_mean = np.mean(specificity, axis=1)
            f1_score_mean = np.mean(f1_score, axis=1)

            macro_avg_precision = np.mean(precision_mean)
            macro_avg_sensitivity = np.mean(sensitivity_mean)
            macro_avg_specificity = np.mean(specificity_mean)
            macro_avg_f1_score = np.mean(f1_score_mean)

            weight_avg_precision = np.sum(
                precision_mean * support[:n_classes]
            ) / np.sum(support[:n_classes])
            weight_avg_sensitivity = np.sum(
                sensitivity_mean * support[:n_classes]
            ) / np.sum(support[:n_classes])
            weight_avg_specificity = np.sum(
                specificity_mean * support[:n_classes]
            ) / np.sum(support[:n_classes])
            weight_avg_f1_score = np.sum(f1_score_mean * support[:n_classes]) / np.sum(
                support[:n_classes]
            )

            precision = precision_mean.tolist() + [
                None,
                macro_avg_precision,
                weight_avg_precision,
            ]
            sensitivity = sensitivity_mean.tolist() + [
                None,
                macro_avg_sensitivity,
                weight_avg_sensitivity,
            ]
            specificity = specificity_mean.tolist() + [
                None,
                macro_avg_specificity,
                weight_avg_specificity,
            ]
            f1_score = f1_score_mean.tolist() + [
                np.mean(accuracy),
                macro_avg_f1_score,
                weight_avg_f1_score,
            ]

            cr_avg_dict = {
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1-score": f1_score,
                "support": support,
            }
            cr_avg = pd.DataFrame.from_dict(cr_avg_dict)
            cr_avg.index = df_index
            write_output_table(
                table=cr_avg,
                table_path=os.path.join(self.summary_dir, f"average_{cr}"),
                index=True,
            )

    def get_roc(self) -> None:
        """
        Aggregate ROC curves across folds and export the mean curves.
        """

        curve_definitions = []

        for task_name in self.config_file.classification_tasks:

            if constants.NUM_CLASSES_PER_LABEL[task_name] == 2:

                curve_definitions.append(
                    {
                        "task_name": task_name,
                        "class_idx": 1,
                        "class_name": None,
                    }
                )

            else:

                for class_idx, class_name in enumerate(
                    constants.CLASS_NAMES_PER_LABEL[task_name]
                ):
                    curve_definitions.append(
                        {
                            "task_name": task_name,
                            "class_idx": class_idx,
                            "class_name": class_name,
                        }
                    )

        for curve_definition in curve_definitions:

            task_name = curve_definition["task_name"]
            class_idx = curve_definition["class_idx"]
            class_name = curve_definition["class_name"]

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            fig = plt.figure()

            for fold in range(self.num_folds):

                fold_path = os.path.join(
                    self.results_dir,
                    f"fold_{fold}",
                )
                metrics_summary = os.path.join(
                    fold_path,
                    constants.RESULTS_DIR_NAME,
                    constants.METRICS_DIR_NAME,
                    f"{constants.PREDICTIONS_SUMMARY_LABELED_STEM}"
                    f"{self.table_extension}",
                )

                if not os.path.isfile(metrics_summary):
                    continue

                metrics_summary_df = read_output_table(metrics_summary, index_col=None)

                labels = np.array(
                    metrics_summary_df[
                        constants.CLASSIFICATION_LABEL_COLUMNS[task_name]
                    ].values
                )

                preds = np.array(
                    metrics_summary_df[
                        constants.CLASSIFICATION_PROBABILITY_COLUMNS[task_name]
                    ].values
                )
                preds = np.array([ast.literal_eval(arr) for arr in preds])

                valid_samples = labels != self.missing_value
                labels = labels[valid_samples]
                preds = preds[valid_samples]

                if constants.NUM_CLASSES_PER_LABEL[task_name] == 2:

                    curve_labels = labels
                    curve_predictions = preds[:, 1]

                else:

                    labels_oh = (
                        LabelBinarizer()
                        .fit(range(constants.NUM_CLASSES_PER_LABEL[task_name]))
                        .transform(labels)
                    )

                    curve_labels = labels_oh[:, class_idx]
                    curve_predictions = preds[:, class_idx]

                fpr, tpr, _ = metrics.roc_curve(
                    y_true=curve_labels,
                    y_score=curve_predictions,
                )

                interpolated_tpr = np.interp(
                    mean_fpr,
                    fpr,
                    tpr,
                )
                interpolated_tpr[0] = 0.0
                tprs.append(interpolated_tpr)

                roc_auc = metrics.auc(fpr, tpr)
                aucs.append(roc_auc)

                plt.plot(
                    fpr,
                    tpr,
                    lw=1,
                    alpha=0.3,
                    label=(f"ROC fold {fold} " f"(AUC = {roc_auc:.2f})"),
                )

            if not tprs:
                plt.close(fig)
                continue

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                lw=2,
                color="r",
                label=constants.ROC_CHANCE_LEVEL_SHORT_LABEL,
                alpha=0.8,
            )

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0

            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)

            plt.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=(
                    rf"Mean ROC " rf"(AUC = {mean_auc:.2f} " rf"$\pm$ {std_auc:.2f})"
                ),
                lw=2,
                alpha=0.8,
            )

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            plt.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                label=r"$\pm$ 1 std. dev.",
            )

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.055, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="best")

            filename_stem = constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                constants.FILENAME_STEM_KEY
            ]

            if class_name is None:
                filename = f"ROC_{filename_stem}{self.image_extension}"
            else:
                class_value = constants.CLASS_VALUES_PER_LABEL[task_name][class_idx]
                filename = f"ROC_{filename_stem}_{class_value}{self.image_extension}"

            plt.savefig(os.path.join(self.summary_dir, filename))
            plt.close(fig)

    def get_pr(self) -> None:
        """
        Aggregate precision-recall curves across folds and export them.
        """
        curve_definitions = []

        for task_name in self.config_file.classification_tasks:
            if constants.NUM_CLASSES_PER_LABEL[task_name] == 2:
                curve_definitions.append((task_name, 1, None))
            else:
                for class_idx, class_name in enumerate(
                    constants.CLASS_NAMES_PER_LABEL[task_name]
                ):
                    curve_definitions.append((task_name, class_idx, class_name))

        for task_name, class_idx, class_name in curve_definitions:
            y_real = []
            y_proba = []
            fig = plt.figure()

            for fold in range(self.num_folds):
                fold_path = os.path.join(self.results_dir, f"fold_{fold}")
                metrics_summary = os.path.join(
                    fold_path,
                    constants.RESULTS_DIR_NAME,
                    constants.METRICS_DIR_NAME,
                    f"{constants.PREDICTIONS_SUMMARY_LABELED_STEM}"
                    f"{self.table_extension}",
                )

                if not os.path.isfile(metrics_summary):
                    continue

                metrics_summary_df = read_output_table(metrics_summary, index_col=None)
                labels = np.array(
                    metrics_summary_df[
                        constants.CLASSIFICATION_LABEL_COLUMNS[task_name]
                    ].values
                )
                preds = np.array(
                    metrics_summary_df[
                        constants.CLASSIFICATION_PROBABILITY_COLUMNS[task_name]
                    ].values
                )
                preds = np.array([ast.literal_eval(arr) for arr in preds])

                valid_samples = labels != self.missing_value
                labels = labels[valid_samples]
                preds = preds[valid_samples]

                if constants.NUM_CLASSES_PER_LABEL[task_name] == 2:
                    curve_labels = labels
                    curve_predictions = preds[:, 1]
                else:
                    labels_oh = (
                        LabelBinarizer()
                        .fit(range(constants.NUM_CLASSES_PER_LABEL[task_name]))
                        .transform(labels)
                    )
                    curve_labels = labels_oh[:, class_idx]
                    curve_predictions = preds[:, class_idx]

                precision, recall, _ = metrics.precision_recall_curve(
                    y_true=curve_labels,
                    y_score=curve_predictions,
                )
                pr_auc = metrics.auc(recall, precision)

                y_real.append(curve_labels)
                y_proba.append(curve_predictions)

                plt.plot(
                    recall,
                    precision,
                    lw=1,
                    alpha=0.3,
                    label="PR fold %d (AUC = %0.2f)" % (fold, pr_auc),
                )

            if not y_real:
                plt.close(fig)
                continue

            y_real = np.concatenate(y_real)
            y_proba = np.concatenate(y_proba)

            precision, recall, _ = metrics.precision_recall_curve(
                y_true=y_real,
                y_score=y_proba,
            )
            pr_auc = metrics.auc(recall, precision)

            plt.plot(
                recall,
                precision,
                color="b",
                label=r"Precision-Recall (AUC = %0.2f)" % pr_auc,
                lw=2,
                alpha=0.8,
            )
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="best")

            filename_stem = constants.CLASSIFICATION_TASK_DEFINITIONS[task_name][
                constants.FILENAME_STEM_KEY
            ]
            if class_name is None:
                filename = f"PR_{filename_stem}{self.image_extension}"
            else:
                class_value = constants.CLASS_VALUES_PER_LABEL[task_name][class_idx]
                filename = f"PR_{filename_stem}_{class_value}{self.image_extension}"

            plt.savefig(os.path.join(self.summary_dir, filename))
            plt.close(fig)

    def get_boxplot(self) -> None:
        """
        Generate boxplots for fold-level Dice score and Hausdorff distance.
        """

        metrics = [
            constants.SEGMENTATION_DICE_SCORE_COLUMN,
            constants.SEGMENTATION_HD_COLUMN,
        ]

        for metric in metrics:

            metric_data = []
            for fold in range(self.num_folds):

                fold_path = os.path.join(self.results_dir, f"fold_{fold}")
                metrics_summary = os.path.join(
                    fold_path,
                    constants.RESULTS_DIR_NAME,
                    constants.METRICS_DIR_NAME,
                    f"{constants.SEGMENTATION_METRICS_STEM}" f"{self.table_extension}",
                )

                if os.path.isfile(metrics_summary):

                    df = read_output_table(metrics_summary, index_col=None)

                    metric_fold = df[metric].values.tolist()

                    metric_data.append(metric_fold)

            plt.figure()
            bp_metrics = plt.boxplot(metric_data, showmeans=True, meanline=True)

            for fold in range(self.num_folds):

                mean_metric = np.mean(metric_data[fold])
                median_metric = np.median(metric_data[fold])

                if metric == constants.SEGMENTATION_DICE_SCORE_COLUMN:

                    if median_metric > mean_metric:
                        plt.text(
                            fold + 1,
                            mean_metric - 0.03,
                            f"{mean_metric:.3f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )
                        plt.text(
                            fold + 1,
                            median_metric + 0.01,
                            f"{median_metric:.3f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )
                    else:
                        plt.text(
                            fold + 1,
                            mean_metric + 0.01,
                            f"{mean_metric:.3f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )
                        plt.text(
                            fold + 1,
                            median_metric - 0.03,
                            f"{median_metric:.3f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )

                else:
                    max_metric = np.max(metric_data[fold])
                    if median_metric > mean_metric:
                        plt.text(
                            fold + 1,
                            mean_metric - 0.03 * max_metric,
                            f"{mean_metric:.2f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )
                        plt.text(
                            fold + 1,
                            median_metric + 0.01 * max_metric,
                            f"{median_metric:.2f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )
                    else:
                        plt.text(
                            fold + 1,
                            mean_metric + 0.01 * max_metric,
                            f"{mean_metric:.2f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )
                        plt.text(
                            fold + 1,
                            median_metric - 0.03 * max_metric,
                            f"{median_metric:.2f}",
                            horizontalalignment="center",
                            fontsize=8,
                        )

            plt.xticks(
                range(1, self.num_folds + 1),
                [f"fold_{fold}" for fold in range(self.num_folds)],
            )
            plt.xlabel("Folds")
            plt.ylabel("Values")
            plt.legend(
                [bp_metrics["medians"][0], bp_metrics["means"][0]], ["median", "mean"]
            )

            if metric == constants.SEGMENTATION_HD_COLUMN:
                plt.savefig(
                    os.path.join(
                        self.summary_dir,
                        f"{constants.BOXPLOT_FILE_STEMS['hd']}"
                        f"{self.image_extension}",
                    )
                )
                plt.close()
            else:
                plt.savefig(
                    os.path.join(
                        self.summary_dir,
                        f"{constants.BOXPLOT_FILE_STEMS['dice']}"
                        f"{self.image_extension}",
                    )
                )
                plt.close()


if __name__ == "__main__":

    does_segmentation = (
        config.model_architecture == constants.CSNET_MODEL_TYPE
        and not config.classification_only
    )

    classification_metrics_kfold = ClassificationMetricsKfold(
        config_file_dir=config_dir,
        results_dir=config.test_results_dir,
    )

    classification_metrics_kfold.get_avg_cr()
    classification_metrics_kfold.get_roc()
    classification_metrics_kfold.get_pr()

    if does_segmentation:
        classification_metrics_kfold.get_metrics_avg()
        classification_metrics_kfold.get_boxplot()
