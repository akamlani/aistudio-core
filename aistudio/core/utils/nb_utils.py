import  textwrap
import  nest_asyncio
from    IPython.display import Image, Markdown, HTML, JSON

from    rich import print as rprint
from    rich.markdown import Markdown
from    rich.progress import track


# %load_ext watermark
# %watermark -v -p numpy,pandas,matplotlib,torch,transformers,datasets -conda

def async_event_handler() -> None:
    "globally patches asyncio to enable event loops to be re-entrant for notebook environments"
    nest_asyncio.apply()

def show_image(path:str, width:int=512, height:int=512):
    display(Image(filename=path, width=width, height=height))

def to_markdown(text: str):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))
