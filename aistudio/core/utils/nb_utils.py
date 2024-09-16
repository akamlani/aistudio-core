import  textwrap
import  nest_asyncio
import  inspect
from    IPython.display import Image, Markdown, HTML, JSON

from    rich import print as rprint
from    rich.markdown import Markdown
from    rich.progress import track


# import warnings
# warnings.filterwarnings('ignore')

# %load_ext watermark
# %watermark -v -p numpy,pandas,matplotlib,torch,transformers,datasets -conda



def async_event_handler() -> None:
    "globally patches asyncio to enable event loops to be re-entrant for notebook environments"
    nest_asyncio.apply()

def show_code(fn:Callable) -> None:
    src_code = inspect.getsource(fn)
    display(HTML(f"<pre>{src_code}</pre>"))

def show_html(text:str) -> None:
    display(HTML(text))

def show_image(path:str, width:int=512, height:int=512) -> None:
    display(Image(filename=path, width=width, height=height))

def to_markdown(text: str):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))
