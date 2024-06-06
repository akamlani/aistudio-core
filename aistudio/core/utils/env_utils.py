import warnings
import subprocess
import logging

from importlib import import_module
from types import ModuleType
from typing import Optional

from pathlib import Path
from appdirs import user_cache_dir
from dotenv import dotenv_values, find_dotenv, load_dotenv


warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)


def read_env(path: str = ".env", verbose: bool = False) -> dict:
    """read dot environment variables

    Args:
        path (str, optional): name of environment in root path. Defaults to ".env".

    Returns:
        dict: key:value of loaded environment variables

    Examples:
    >>> read_env(".env.shared")
    >>> read_env()
    """
    try:
        load_dotenv(find_dotenv(filename=path, raise_error_if_not_found=True), verbose=verbose)
        config: dict = dotenv_values(path)
        return config
    except Exception as e:
        logger.exception(f"Exception Occured Reading DotFile:{e}")


def read_root_dir() -> str:
    """Retrieve root package dir via git shell command

    Returns:
        str: root directory path
    """
    return subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, text=True).stdout.strip()


def read_cache_dir(author: str = None, app: str = None) -> str:
    """Retrieves and creates the cache dir

    Args:
        author (str): author name to associate with cache directory, only used in windows.
        app (str): project or applicaiton name of applicaition dir

    Returns:
        str: cache directory location

    Examples:
    >>> get_cache_dir('akamlani', Path('huggingface').joinpath('hub'))
    """
    cache_dir = user_cache_dir(app, author)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir
