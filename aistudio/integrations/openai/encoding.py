import  numpy as np
import  pandas as pd
import  tiktoken
from    typing import List, Dict, Generator, Any

class Tokenizer(object):
    """Tokenizer for OpenAI models

    Reference Links:
    https://platform.openai.com/tokenizer

    data = {
        "[BOS]": "(beginning of sequence) marks the beginning of text", 
        # equivalent to <|endoftext|>
        # <|endoftext|> for padding (as we typically use a mask when training on batched inputs)
        "[EOS]": "(end of sequence) marks where the text ends (this is usually used to concatenate multiple unrelated texts",
        "[PAD]": "for batch size > 1, may include multiple texts with different lengths.  padding shorter texts to longest length for equal length.",
        # e.g., GPT-2 uses a BPE tokenizer, so no [UNK], which breaks down the words into subword units
        "[UNK]": "words that are not included in the vocabulary"
    }

    Examples:
    >>> tok=Tokenizer(model_name='gpt2')
    # over string of text 
    >>> ids:List[int]     = tok.encode(raw_text,  allowed_special={"<|endoftext|>"})
    >>> decoded:List[str] = tok.decode(ids)
    # over a dataframe 
    >>> df_tok_trsfrm     = tok.pipe(df)
    # calculate statistics over a dataframe 
    >>> from    aistudio.datasets.info import InfoText
    >>> df_tok_stats      = InfoText.calc_record_statistics(df_tok_trsfrm)
    >>> df_frm_stats      = InfoText.calc_frame_statistics(df_tok_stats)
    display(df_tok_stats)
    display(df_frm_stats)
    """
    def __init__(self, model_name:str="gpt-4"):
        self.encoding   = tiktoken.encoding_for_model(model_name)
        self.name       = self.encoding.name
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: List[str], allowed_special:set={"<|endoftext|>"}, **kwargs) -> List[int]:
        return self.encoding.encode(text, allowed_special=allowed_special, **kwargs)

    def decode(self, ids: List[int]) -> List[str]:
        return self.encoding.decode(ids, errors="strict")

    def decode_encoding(self, ids: List[int]) -> Dict[int, str]:
        return {token:self.encoding.decode([token]) for token in ids}

    def pipe(self, df: pd.DataFrame, col: str='text', **kwargs) -> pd.DataFrame:
        return df.assign(
            words       = lambda df_: df_[col].apply(lambda s:      s.split()),
            tokens      = lambda df_: df_[col].apply(lambda s:      self.encode(s, **kwargs)),
            decoded     = lambda df_: df_['tokens'].apply(lambda x: self.decode(x)),
            token_map   = lambda df_: df_['tokens'].apply(lambda x: self.decode_encoding(x))
        )
