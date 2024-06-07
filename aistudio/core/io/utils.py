import pandas as pd
import re

import logging

logger = logging.getLogger(__name__)


def trsfrm_col_camelcase_to_snakecase(col: str) -> str:
    """Transforms column naming from camelcase to snakecase

    Args:
        col (str): input column name to transfrom

    Returns:
        str: transformed column
    """
    column = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", col)
    column = re.sub("([a-z0-9])([A-Z])", r"\1_\2", col).lower()
    return column.replace(" ", "_")


def trsfrm_frame_camelcase_to_snakecase(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms column naming from camelcase to snakecase for a dataframe

    Args:
        df (pd.DataFrame): input dataframe with columns

    Returns:
        pd.DataFrame: transformed pandas dataframe
    """
    df.columns = map(trsfrm_col_camelcase_to_snakecase, df.columns)
    return df


def trsfrm_normalize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """renames columns given a dictionary mapping

    Args:
        df (pd.DataFrame): input frame
        mapping (dict): maaping from dictionary in src:dest format

    Returns:
        pd.DataFrame: _description_
    """
    return df.rename(columns=mapping) if mapping else df
