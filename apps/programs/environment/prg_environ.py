import  os
from    dotenv import load_dotenv
from    rich.console import Console

def load_secrets():
    load_dotenv()

    Console().print(f"""Loading secrets...
        {os.environ["OPENAI_API_KEY"]},
        {os.environ["HUGGINGFACEHUB_API_TOKEN"]}
    """)


if __name__ == "__main__":
    script_path = os.path.dirname(__file__)
    load_secrets()