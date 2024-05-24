import numpy  as np
import random
import logging

from   typing import List
from   pathlib import Path
from   omegaconf import DictConfig
from   appdirs import user_cache_dir

from   .io.reader import read_hydra
from   .io.filesystem import get_username
from   .utils.env_utils import read_env, read_root_dir

logger = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, root_path:str=None):
        self.seed_init()
        # directories
        self.root_dir        = Path(read_root_dir()) if not root_path else root_path
        self.conf_dir        = self.root_dir.joinpath(Path('config'))
        self.data_dir        = self.root_dir.joinpath(Path('data'))
        self.cache_dir       = user_cache_dir()
        # user inforamtion
        self.username        = get_username()
        # load environment
        self.env:dict        = read_env('.env')

    def register_catalog(self, active_catalog:str) -> Path:
        self.catalog_dir     = active_catalog
        return Path(self.catalog_dir)

    def create(self, experiment_name:str, tags:List[str], **kwargs) -> DictConfig:
        clone:DictConfig = read_hydra(self.conf_dir.joinpath('experimentation', 'experiment.yaml'))
        clone.experiment.name = experiment_name
        clone.experiment.tags = tags
        clone.experiment.project.name = experiment_name
        clone.experiment.project.tags = tags
        clone.experiment.install.dir  = str( self.root_dir.joinpath(clone.experiment.install.dir) )

        # create directories
        Path(clone.experiment.project.path).mkdir(parents=True, exist_ok=True)
        return clone

    def seed_init(self, seed:int=42) -> int:
        random.seed(seed)
        np.random.seed(seed)
        return seed
