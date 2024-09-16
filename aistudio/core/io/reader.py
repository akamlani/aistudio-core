import pandas as pd
import yaml
import json
import logging
import functools
import urllib.request

from pathlib import Path
from typing import TypeVar, Callable, Optional, List, Dict, Tuple, Any
from omegaconf import DictConfig, OmegaConf

from .filesystem import is_valid_url

logger = logging.getLogger(__name__)


# define potential return types for the decorating function
T = TypeVar("T", pd.DataFrame, DictConfig, Dict[str, Any], List[Any], Any, None)

dwld_file     = lambda src_uri, dst_name: urllib.request.urlretrieve(src_uri, dst_name)
hydra_to_yaml = lambda cfg: OmegaConf.to_yaml(cfg)

def read_content(filename:str, ext='.csv') -> pd.DataFrame:
    with open(filename, 'r') as f:
        csv_reader = csv.DictReader(f)
        content    = [row for row in csv_reader]
        return pd.DataFrame(content)

def read_remote_content(uri:str) -> dict:
    import requests 
    return requests.get(uri).content



def read_exec_io(
    func: Callable[[str, Tuple[Any, ...], Dict[str, Any]], T],
) -> Callable[[str, Tuple[Any, ...], Dict[str, Any]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(path: str, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[T]:
        try:
            if Path(path).is_file() or is_valid_url(path):
                return func(path, *args, **kwargs)
            else:
                logger.error(f"Path: {path} is malformed or does not Exist")
                return None
        except Exception as e:
            logger.exception(f"Exception Occured Reading File: {path}:{e}")
            return None

    return wrapper


@read_exec_io
def read_yaml(filepath: str, encoding: str = "utf-8", *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[dict]:
    """Reads a Yaml file for usage primarily with configuration

    Args:
        filepath (str): path to location of file to read
        encoding (str, optional): encoding representation of file. Defaults to "utf-8".

    Returns:
        Optional[dict]: dictionary of contents read
    """
    with open(filepath, encoding=encoding) as f:
        data: dict = yaml.load(f, Loader=yaml.FullLoader)
        return data


@read_exec_io
def read_hydra(filepath: str, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[DictConfig]:
    """Reads a Hyrda Yaml Configuraiton

    Args:
        filepath (str): path to location of file to read

    Returns:
        Optional[DictConfig]: Structured Dictionary of OmegaConf
    """
    return OmegaConf.load(filepath)


@read_exec_io
def read_json(filepath: str, encoding: str = "utf-8", *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[dict]:
    """reads a JSON file

    Args:
        filepath (str): path to location of file to read
        encoding (str, optional): encoding representation of file, default to utf-8. Defaults to "utf-8".

    Returns:
        Optional[dict]: dictionary of contents read from file
    """
    with open(filepath, encoding=encoding) as f:
        data: dict = json.load(f)
        return data


@read_exec_io
def read_json_to_pandas(
    filepath: str, lines: bool = False, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Reads lines encoded in JSON format (*.jsonl) into Pandas format

    Args:
        filepath (str): path to location of file to read

    Returns:
        Optional[pd.DataFrame]: pandas dataframe of jsone lines format read
        lines (bool, optional): records or lines format. Defaults to False.

    Examples:
    >>> df_dataset = .read_jsonl_to_pandas(filepath="docs.jsonl", lines=True)
    >>> examples   = df_dataset.to_dict()
    >>> df_dataset.shape
    """
    return pd.read_json(filepath, orient="records", lines=lines, **kwargs)


@read_exec_io
def read_excel_to_pandas(
    filepath: str, sheet_name: str = "Sheet1", *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """Reads a CSV file

    Args:
        filepath (str): path to location of file to read
        sheetname (str): name of sheet to read

    Returns:
        Optional[pd.DataFrame]: returns a pandas dataframe if successful

    Examples:
    >>> read_excel_to_pandas("./customers.csv", sheet_name='customers).pipe(trsfrm_camelcase_to_snakecase)
    """
    return pd.ExcelFile(filepath, *args, **kwargs).parse(sheet_name)


@read_exec_io
def read_csv_to_pandas(filepath: str, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Reads a CSV file

    Args:
        filepath (str): path to location of file to read

    Returns:
        Optional[pd.DataFrame]: returns a pandas dataframe if successful

    Examples:
    >>> read_csv_to_pandas("./vendors.csv", parsed_dates=['Date']).pipe(trsfrm_camelcase_to_snakecase)
    """
    return pd.read_csv(
        filepath,
        *args,
        sep=kwargs.get("sep", ","),  # e.g., ['\s+', '\t', ',']
        **kwargs,
    )


@read_exec_io
def read_text(filepath: str, encoding: str = "utf-8", *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[str]:
    """Reads a text file

    Args:
        filepath (str): path to location of file to read
        encoding (str, optional): encoding representation of file, default to utf-8. Defaults to "utf-8".

    Returns:
        Optional[str]: returns a string if successful
    """
    with open(filepath, encoding=encoding) as f:
        return f.read()
