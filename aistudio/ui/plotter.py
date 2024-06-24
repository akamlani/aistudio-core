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
