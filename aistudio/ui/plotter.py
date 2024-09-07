import numpy as np
import pandas as pd
from   typing import List

import matplotlib.pyplot as plt
import seaborn as sns

def plot_image(image_array:np.array, cmap:str='gray') -> None:
    """Plot image from input array.

    Args:
        image_array (np.array): input image array
        cmap (str, optional): color map to use. Defaults to 'gray'.

    Examples:
    >>> plot_image(image_array, cmap='gray')
    """
    plt.imshow(image_array, cmap=cmap)
    plt.axis('off')


def plot_heatmap(labels:List[str], emb:List[np.ndarray], rotation:int=90) -> None:
    """ Plot Heatmap

    Examples:
    >>> question_embedding = model.encode(question)
    >>> answers_embeddings = model.encode(answers)
    >>> df_emb = get_similar(
        question_embedding, 
        answers_embeddings, 
        answers, 
        top_k=len(answers_embeddings)
    )
    >>> df_answers = pd.DataFrame(
        zip(answers, answers_embeddings), 
        columns=['text', 'emb']
    )
    >>> sim_matrix = cosine_similarity(answers_embeddings)
    >>> df_sim     = pd.DataFrame(
            sim_matrix, 
            index   = df_emb['text'], 
            columns = df_emb['text']
        )
    >>> plot_similarity(labels=df_sim.index, emb=df_sim.values, rotation=90)
    """
    sns.set_theme(font_scale=1.2)
    g = sns.heatmap(emb, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    # return g