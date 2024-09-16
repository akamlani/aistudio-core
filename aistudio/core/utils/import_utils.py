import  importlib
from    importlib import import_module
from    importlib.metadata import version #version(f"{library_name}")
from    typing import Any, Optional
from    types import ModuleType

import logging
logger = logging.getLogger(__name__)



def import_modules(module_path:str, class_name:str) -> Any:
    """dynamically import a module and class

    Args:
        module_path (str): path to module
        class_name (str): path to class name or function in module

    Returns:
        Any: reference to imported class or function

    Equivalent to the following example:
    >>> try:
            import  torch
            import  transformers
            import  datasets
        except ImportError:
            raise ImportError(
                "Please install the following packages: torch, transformers, datasets"
            )
    """
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        msg = f"Missing optional dependency: '{class_name}'. " f"Use poetry or pip to install '{class_name}'."
        raise ModuleNotFoundError(msg)
        return None


def import_optional_dependency(name: str) -> Optional[ModuleType]:
    """Import an optional dependency.

    If a dependency is missing, an ImportError with a nice message will be raised.

    Args:
        name (str): module name

    Raises:
        ImportError: unable to import module

    Returns:
        Optional[ModuleType]: The imported module when found
    """
    msg = f"Missing optional dependency: '{name}'. " f"Use poetry or pip to install '{name}'."
    try:
        module = import_module(name)
    except ImportError:
        raise ImportError(msg)

    return module
