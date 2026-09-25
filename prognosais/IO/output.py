"""Utilities for reading and writing generated output tables."""

from pathlib import Path

import pandas as pd

from prognosais.IO import constants


def read_output_table(
    table_path: str | Path,
    index_col: int | str | None,
) -> pd.DataFrame:
    """Read a generated CSV or Excel table according to its extension.

    Args:
        table_path: Path to a generated table.
        index_col: Column to use as dataframe index, or None to retain all
            columns as data.

    Returns:
        Parsed dataframe.

    Raises:
        ValueError: If the table extension is not supported.
    """
    path = Path(table_path)
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=index_col)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, index_col=index_col)
    raise ValueError(
        f"Unsupported output table extension '{path.suffix}' for {path}. "
        "Supported values: " + ", ".join(constants.SUPPORTED_TABLE_OUTPUT_FORMATS)
    )


def write_output_table(
    table: pd.DataFrame,
    table_path: str | Path,
    index: bool,
) -> None:
    """Write a generated dataframe as CSV or Excel according to its extension.

    Args:
        table: Dataframe to persist.
        table_path: Destination path whose extension selects the file format.
        index: Whether to include the dataframe index in the output file.

    Raises:
        ValueError: If the table extension is not supported.
    """
    path = Path(table_path)
    if path.suffix == ".csv":
        table.to_csv(path, index=index)
        return
    if path.suffix == ".xlsx":
        table.to_excel(path, index=index)
        return
    raise ValueError(
        f"Unsupported output table extension '{path.suffix}' for {path}. "
        "Supported values: " + ", ".join(constants.SUPPORTED_TABLE_OUTPUT_FORMATS)
    )
