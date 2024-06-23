import  urllib.request
from    typing import Optional 
from    pathlib import Path 
from    omegaconf import DictConfig, OmegaConf

from    .filesystem import get_uri_properties

def download_file(uri:str, export_dir:Optional[str]=None) -> str:
    properties:DictConfig = get_uri_properties(uri)
    export_path = Path(export_dir).joinpath(properties.filename)     
    urllib.request.urlretrieve(uri, str(export_path))
    return str(export_path)
