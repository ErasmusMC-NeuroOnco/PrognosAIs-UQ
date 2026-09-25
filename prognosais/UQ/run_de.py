"""Command-line runner for deterministic deep ensemble aggregation."""

from __future__ import annotations

import argparse

from prognosais.IO import constants
from prognosais.UQ.de import run_de


def validate_args(args: argparse.Namespace) -> None:
    """Validate that every required CLI parameter was explicitly passed.

    Args:
        args: Parsed command-line namespace.

    Raises:
        ValueError: If a required value is missing or empty.
    """

    required_fields = (
        "ensemble_root",
        "seeds",
        "output_dir",
        "tasks",
        "seed_folder_template",
        "dropout_rate",
        "predictions_relative_path",
        "inference_mode",
        "probability_map_relative_path",
        "missing_label",
        "probability_sum_tolerance",
        "entropy_epsilon",
        "include_segmentation",
        "table_extension",
    )
    for field in required_fields:
        value = getattr(args, field)
        if value is None:
            raise ValueError(f"Missing required CLI parameter: {field}.")
        if isinstance(value, str) and value.strip() == "":
            raise ValueError(f"Missing required CLI parameter: {field}.")
        if isinstance(value, list) and len(value) == 0:
            raise ValueError(f"Missing required CLI parameter: {field}.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DE aggregation.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate deterministic predictions from independently trained "
            "models into deep ensemble uncertainty outputs."
        )
    )
    parser.add_argument(
        "--table-extension",
        required=True,
        choices=constants.SUPPORTED_TABLE_OUTPUT_FORMATS,
        help="Configured extension for generated tables.",
    )
    parser.add_argument(
        "--ensemble-root",
        required=True,
        help=(
            "Root directory containing the ensemble seed folders, for example "
            "dropout_025_seed_{seed}."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed folders to include as ensemble members.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where DE UQ outputs are written.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(constants.CLASSIFICATION_TASK_DEFINITIONS),
        required=True,
        help="Classification tasks to process.",
    )
    parser.add_argument(
        "--seed-folder-template",
        required=True,
        help="Template used to locate each seed folder.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        required=True,
        help="Dropout-rate token used in DE segmentation output filenames.",
    )
    parser.add_argument(
        "--predictions-relative-path",
        required=True,
        help=(
            "Relative path from each seed folder to the prediction summary "
            "selected by --inference-mode."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=constants.SUPPORTED_INFERENCE_MODES,
        required=True,
        help="Whether the selected prediction summaries are labeled or unlabeled.",
    )
    parser.add_argument(
        "--probability-map-relative-path",
        required=True,
        help=(
            "Relative path template from each seed folder to a case probability "
            "map. Must contain the {case} field."
        ),
    )
    parser.add_argument(
        "--missing-label",
        type=int,
        required=True,
        help="Label value indicating unavailable annotations.",
    )
    parser.add_argument(
        "--probability-sum-tolerance",
        type=float,
        required=True,
        help="Absolute tolerance for probability vectors summing to one.",
    )
    parser.add_argument(
        "--entropy-epsilon",
        type=float,
        required=True,
        help="Numerical-stability epsilon used when computing entropy.",
    )
    segmentation_group = parser.add_mutually_exclusive_group(required=True)
    segmentation_group.add_argument(
        "--include-segmentation",
        action="store_true",
        dest="include_segmentation",
        help="Aggregate saved segmentation probability maps.",
    )
    segmentation_group.add_argument(
        "--skip-segmentation",
        action="store_false",
        dest="include_segmentation",
        help="Only aggregate classification outputs.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    """Run DE aggregation and print a concise output summary."""

    args = parse_args()
    outputs = run_de(
        ensemble_root=args.ensemble_root,
        seeds=args.seeds,
        output_dir=args.output_dir,
        tasks=args.tasks,
        dropout_rate=args.dropout_rate,
        seed_folder_template=args.seed_folder_template,
        predictions_relative_path=args.predictions_relative_path,
        inference_mode=args.inference_mode,
        probability_map_relative_path=args.probability_map_relative_path,
        missing_label=args.missing_label,
        probability_sum_tolerance=args.probability_sum_tolerance,
        include_segmentation=args.include_segmentation,
        entropy_epsilon=args.entropy_epsilon,
        table_extension=args.table_extension,
    )

    print("Deep ensemble aggregation completed.")
    print("Summary:")
    print(outputs.summary_table.to_string(index=False))
    print("Output files:")
    for name, path in outputs.output_paths.items():
        print(f"  {name}: {path}")

    if outputs.exclusion_table.empty:
        print("No exclusions were recorded.")
    else:
        print("Exclusions by reason:")
        print(
            outputs.exclusion_table.groupby([constants.TASK_COLUMN_NAME, "Reason"])
            .size()
            .reset_index(name="count")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
