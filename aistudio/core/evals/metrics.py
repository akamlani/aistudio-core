import  numpy  as np
import  pandas as pd
from    typing import List

# euclidean:            for non-normalized vectors
# hamming:              hamming distance on binary vectors, e.g., BPR, (considerably cheaper than calculating the cosine distances)
# dot product:          similarity unnormalized between two vectors, magnitude of the projection (orientation) of one vector onto the other
# cosine distance:      measuring difference in direction between two vectors
# cosine similarity:    normalized dot product, when we have normalized vectors




calc_accuracy_binary   = lambda yhat: sum(yhat)/len(yhat)
calc_accuracy          = lambda ytrue, yhat: sum(x == y for x, y in zip(yhat, ytrue))/len(yhat)

calc_l2_distance       = lambda v1, v2: (v1 - v2) ** 2
calc_l1_distance       = lambda v1, v2: (v1 - v2)
calc_cosine_similarity = lambda v1, v2: (np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
calc_cosine_distance   = lambda v1, v2: 1 - calc_cosine_similarity(v1, v2)
calc_dot_prod_ip       = lambda v1, v2: np.dot(v1, v2)                     # ~ np.sum(Ai*bi)
trsfrm_norm_vector     = lambda v: v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

calc_euclidean_l2      = lambda v1, v2: np.linalg.norm((v1 - v2), ord=2)   # ~ np.sqrt(np.array(L2)).sum())
calc_manhattan_l1      = lambda v1, v2: np.linalg.norm((v1 - v2), ord=1)   # ~ np.abs(np.array(L1)).sum()


# distances
l1_norm                = lambda x: np.sum(map(np.abs, x))
l2_norm                = lambda x: np.linalg.norm(x)
l2_norm_sq             = lambda x: np.dot(x, x)


def sort_scores(scores:List[float]) -> List[float]:
    """sort scores

    Args:
        scores (List[float]): scoring, e.g., from similarities

    Returns:
        List[float]: scores for each index

    Examples:
    >>> similarities = embedding.dot(model.encode("ESPN"))
    >>> sorted_idx   = sort_scores(similarities)
    >>> mappings     = {k:similarities[k] for k in sorted_idx[:5]}
    """
    sorted_ix = np.argsort(-scores)
    return sorted_ix

def jaccard_similarity(v1:np.array, v2:np.array) -> float:
    """Calculates the Jaccard similarity coefficient (IOU) between two sets.

    Identify common elements between sets and quantify extent of overlap,
    commonly when dealing with binary or categorical data.

    Args:
        v1 (np.array): The first set.
        v2 (np.array): The second set.

    Returns:
        float: The Jaccard similarity coefficient between two inputs.

    Examples:

    def calc_jaccard_similarity(set1:set, set2:set) -> float:
        set1, set2   = set(set1), set(set2)
        intersection = len(set1.intersection(set2))
        union        = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    >>> v1    = np.array({1, 2, 3, 4})
    >>> v2    = np.array({3, 4, 5, 6})
    >>> score = jaccard_similarity(v1, v2)
    """
    intersection = np.logical_and(v1, v2)
    union        = np.logical_or(v1, v2)

    try:
        return np.sum(intersection) / np.sum(union)
    except ZeroDivisionError:
        return 0


def calc_embedding_metrics(emb1:np.ndarray, emb2:np.ndarray) -> pd.DataFrame:
    """get embedding metrics

    Args:
        emb1 (np.ndarray): input embedding vector
        emb2 (np.ndarray): input embedding vector

    Returns:
        pd.DataFrame: computed metrics
    """
    return pd.DataFrame([dict(
        manhattan_l1        = calc_manhattan_l1(emb1, emb2),
        euclidean_l2        = calc_euclidean_l2(emb1, emb2),
        jaccard_similarity  = jaccard_similarity(emb1, emb2),
        dot_product         = calc_dot_prod_ip(emb1, emb2),
        cosine_similarity   = calc_cosine_similarity(emb1, emb2),
        cosine_distance     = calc_cosine_distance(emb1, emb2),
    )], index=['metrics']).style.format("{:.3f}")




