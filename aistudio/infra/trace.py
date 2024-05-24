import logging
import logging.config
from pathlib import Path
from typing import Any, Dict

import coloredlogs
from omegaconf import DictConfig, OmegaConf

from ..core.io.reader import read_hydra


def configure_logging(file_path: str, prefix: str = None, level: int = None, **kwargs) -> None:
    """
    Perform logging configuration
    :param file_path: file path where logging file is located (*.logging.yaml)
    :param prefix: overrided path to where file logging should be performed
    :param level: corresponding integer level to override coloredlogs module with
    :return:
    """
    # read logging configuration
    logger_config: DictConfig = read_hydra(file_path)
    # determine if we should overide default logging path
    output = Path(logger_config.handlers.file_handler.filename)
    output = Path(prefix).joinpath(output.name) if prefix else output
    logger_config.handlers.file_handler.filename = str(output)
    Path(output.parent).mkdir(parents=True, exist_ok=True)
    # configure logger, create logger/formatter to the handler via dictConfig(*.yaml) instead of fileConfig(*.conf)
    logger_config: Dict[str, Any] = OmegaConf.to_container(logger_config, resolve=True)
    logging.config.dictConfig(logger_config)
    # default coloredlogs: TBD check if RichHandler not enabled
    coloredlogs.install(level=level) if level else coloredlogs.install()
