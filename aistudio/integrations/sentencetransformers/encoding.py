import  numpy  as np
import  pandas as pd
import  torch
from    typing import List, Optional, Union
from    sentence_transformers import SentenceTransformer, CrossEncoder, util

class DenseEncoder(object):
    def __init__(self, model_id:str= "sentence-transformers/all-mpnet-base-v2", max_seq_len:int=256):
        """Creates a semantic dense encoder

        Dimensiona and models:
        384 dim: "paraphrase-MiniLM-L6-v2"
                  https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        384 dim: "multi-qa-MiniLM-L6-cos-v1"
        768 dim: "sentence-transformers/all-mpnet-base-v2"
                  https://huggingface.co/sentence-transformers/all-mpnet-base-v2

        "sentence-transformers/all-MiniLM-L12-v2"
        "sentence-transfromers/all-MiniLM-L6-v2"
        Args:
            model_id (str, optional): model to use. Defaults to 'sentence-transformers/all-mpnet-base-v2'.
            max_seq_len (int, optional): maximum sequence length, truncation. Defaults to 256.
            device (str, optional): device as cpu or gpu. Defaults to 'cpu'.
        """
        self.device_gpu:bool        = torch.cuda.is_available()
        self.device:torch.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len:int        = max_seq_len
        self.encoder                = SentenceTransformer(model_id).to(self.device)
        self.encoder.max_seq_length = self.max_seq_len

    def fit(self, texts:List[str], convert_to_tensor:bool=True, show_progress:bool=True) -> torch.Tensor:
        self.data:List[str] = texts if isinstance(texts, list) else [texts]
        return self.encoder.encode(self.data, convert_to_tensor=convert_to_tensor, show_progress_bar=show_progress)

    def get_encoded_lengths(self, texts:List[str]):
        return [self.encode(text)['input_ids'].shape[1] for text in texts]

    def normalize_l2(self, embeddings:np.ndarray) -> np.ndarray:
        # L2 Normalize the rows
        return embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))

    def encode(self,
        text:Union[str, np.array, List[str]],
        batch_size:int=128,
        normalize_embeddings:bool=True,
        convert_to_tensor:bool=True,
        show_progress:bool=True,
    ) -> torch.Tensor:
        emb = self.encoder.encode(
            text,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            device=self.device,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=show_progress
        )
        # emb = emb.tolist()
        return emb if not self.device_gpu else emb.cuda()

    def score(self,
        query_emb:Union[torch.Tensor, np.array, list],
        corpus_emb:Union[torch.Tensor, np.array, list],
        top_k:int=100
    ) -> pd.DataFrame:
        # top_k: number of records to retrieve
        hit_at_k:List[List[dict]] =  util.semantic_search(query_emb, corpus_emb, top_k=top_k)
        return pd.DataFrame(hit_at_k[-1]).round(3)

    def lookup(self, df_scores:pd.DataFrame, df_emb:pd.DataFrame) -> pd.DataFrame:
        # for a particular individual query
        return (
            df_emb
            .join(df_scores)
            .sort_values(by='score', ascending=False)
            .rename(columns={'score':'dense_score'})
        )


class Ranker(object):
    def __init__(self, model_name:str='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """Reranks the candidates based on the query using a cross-encoder

        Args:
            model_name (str, optional): model to use. Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
            # 'cross-encoder/stsb-distilroberta-base'
            # 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            # 'cross-encoder/ms-marco-MiniLM-L-12-v2'

        # https://www.reddit.com/r/huggingface/comments/16snrpo/bgereranker_how_to_use/
        # https://www.sbert.net/examples/applications/cross-encoder/README.html
        Examples:
        >>> ranker     = Ranker()
        >>> top_k_rank = 100
        >>> hit_at_k   = ranker.rank(
            "What is the capital of the United States?",
            list(df_candidates_dense['text_small'][:top_k_rank])
        )
        >>> df_candidates_reranked = ranker.lookup(df_candidates_dense[:top_k_rank], hit_at_k)
        >>> df_candidates_reranked.shape, df_candidates_reranked[df_candidates_reranked.cross_score.notna()].shape
        """
        self.encoder = CrossEncoder(model_name)

    def rank(self, query:str, candidates:List[str]) -> List[float]:
        # score all the retrieved passages with cross encoder
        sz        = len(candidates)
        queries   = [query] * sz
        cross_inp = list(zip(queries, candidates))
        scores    = self.encoder.predict(cross_inp)
        return scores

    def lookup(self, df_src:pd.DataFrame, scores:np.ndarray) -> pd.DataFrame:
        return (
            df_src
            .assign(cross_score=scores)
            .sort_values(by='cross_score', ascending=False)
        )
