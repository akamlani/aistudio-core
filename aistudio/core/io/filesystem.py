import pandas as pd

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from omegaconf import DictConfig

import logging

logger = logging.getLogger(__name__)


create_dirs  = lambda uri: Path(uri).mkdir(exist_ok=True, parents=True)
get_username = lambda: Path.home().name


def search_files(uri: str, extension: str = None) -> List[str]:
    """recursively searches for files matching the extension

    Args:
        uri (str): path to directory
        extension (str): extension pattern to search for. Defaults to None.

    Returns:
        List[str]: list or string uri's
    """
    items = map(str, Path(uri).rglob("*" + extension if extension else "*"))
    return sorted(filter(lambda x: Path(x).is_file(), items))


def search_files_to_dataframe(uri: str, extension: str = None) -> pd.DataFrame:
    """generates a dataframe of recurivesly found files matching extension

    Args:
        uri (str): path to directory
        extension (str): extension pattern to search for. Defaults to None.

    Returns:
        pd.DataFrame: dataframe with full uri and separate filename attributes
    """
    df = (
        pd.DataFrame(search_files(uri, extension), columns=["uri"])
        .assign(
            filename=lambda df_: df_.uri.apply(lambda uri: Path(uri).name),
            name=lambda df_: df_.filename.apply(lambda x: Path(x).stem),
        )
        .sort_values(by=["filename"], ascending=True)
        .reset_index(drop=True)
    )[["name", "filename", "uri"]]
    return df


def list_matching_files(parent_dir: str, pattern: str) -> List[str]:
    """list matching files in a directory from a pattern

    Args:
        parent_dir (str): parent directory to seed from
        pattern (str): pattern to match

    Returns:
        List[str]: list of files that matched the pattern

    Examples
    >>> list_matching_files(exp_data_export_dir, os.path.join("train", "*.parquet"))
    """
    return sorted(parent_dir.glob(pattern))


def get_uri_properties(uri: str, root_dir: Optional[str] = None) -> DictConfig:
    """get properties of a uri, either as a local or remote resource

    Args:
        uri (str): path to directory
        root_dir (Optional[str], optional): root directory for local files. Defaults to None.

    Returns:
        DictConfig: Hydra DictConfig of properties
    """
    is_remote = lambda s: is_url_remote(s) and is_valid_url(s)
    remote_state = is_remote(uri)
    uri = uri if remote_state else f"{root_dir}/{uri}" if not Path(uri).exists() else uri

    is_dir = Path(uri).is_dir()
    ext_suffix = Path(uri).suffix if not is_dir else Path(uri).glob("**/*").__next__().suffix
    properties = dict(
        uri=uri,
        name=Path(uri).stem,
        filename=Path(uri).name,
        is_remote=remote_state,
        is_file=Path(uri).is_file() if not remote_state else ext_suffix != "",
        is_dir=is_dir,
        ext_suffix=ext_suffix,
        ext_name=ext_suffix.lstrip("."),
    )
    return DictConfig(properties)


def is_valid_url(uri: str) -> bool:
    """Validates if url is valid

    Args:
        uri (str): url

    Returns:
        bool: validation performance
    """
    try:
        result = urlparse(uri)
        # check for a limited version of supported formats
        return result.scheme in ["http", "https"] and all([result.scheme, result.netloc])
    except ValueError as e:
        logger.exception(f"Exception Occured Reading url: {uri}:{e}")
        return False


def is_url_remote(uri: str) -> bool:
    """Check if the uri is local filesystem or remote

    Args:
        uri (str): uri to parse

    Returns:
        bool: returns True if remote url
    """
    try:
        result = urlparse(uri)
        return True if result.scheme and result.scheme != "file" else False
    except ValueError as e:
        logger.exception(f"Exception Occured Reading url: {uri}:{e}")
        return False
