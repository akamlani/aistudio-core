import logging
from pathlib import Path
from omegaconf import DictConfig
from rich.logging import RichHandler

from .core.io.reader import read_hydra
from .infra.trace import configure_logging
from .version import watermark

root_logger = logging.getLogger("root")


def setup_configure() -> DictConfig:
    """
    Default setup configuration upon module load
    :return:
    """
    # load experiment structure to be installed at root
    select_path: str = Path(__file__).parent.joinpath("..").resolve()
    conf_folder: str = select_path.joinpath("config").resolve()
    # gather experiment structure to be installed at root
    exp_conf: DictConfig = read_hydra(conf_folder.joinpath("experimentation", "experiment.yaml"))
    exp_log_dir: str = exp_conf.experiment.reporting.logs
    exp_log_dir: str = select_path.joinpath(exp_log_dir).joinpath("..").resolve()
    # !!!configure logging with prefix from associated experiment path
    log_conf_path: str = Path(conf_folder).joinpath("logging.yaml")
    configure_logging(file_path=log_conf_path, prefix=exp_log_dir)
    return exp_conf
