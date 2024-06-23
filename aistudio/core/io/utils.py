import pandas as pd
import base64 
import requests 
import re

from   .filesystem import is_url_remote

import logging

logger = logging.getLogger(__name__)



def trsfrm_to_base64(uri: str) -> str:
    """Transforms a file, local or remote uri, to a base64 encoded string

    Args:
        uri (str): uri to asset, e.g., uri to image or video location

    Returns:
        str: transformed base64 encoded string

    Examples:
    >>> for batch in loader:
            for name in source:
                    batch.add_object({
                            "name": name,            # name of the file
                            "path": path,            # path to the file to display result
                            "image": toBase64(path), # this gets vectorized - "image" was configured in vectorizer_config as the property holding images
                            "mediaType": "image",    # a label telling us how to display the resource 
                        })
    """
    is_remote = is_url_remote(uri)
    if not is_remote:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    else: 
        response = requests.get(url)
        content  = response.content
        return base64.b64encode(content).decode('utf-8')


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
