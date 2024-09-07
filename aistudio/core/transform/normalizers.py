import numpy as np
import pandas as pd
from   typing import List, Tuple 

# mu, std = get_data_statistics(X_train)
# X_train_sc = .trsfrm_scaler(X_train, mu, std)
# X_test_sc  = .trsfrm_scaler(X_test, mu, std)
trsfrm_scaler = lambda X, mu, std: (X - mu) / std

processor_min_max_trsfrm        = lambda xs: (xs - xs.min()) / (xs.max() - xs.min())    # maintain disribution, but for smaller scale 
processor_standarization_trsfrm = lambda xs: (xs - xs.mean()) / xs.std()                # centered and scaled
processor_log_trsfrm            = lambda xs: np.log1p(xs)                               # for skewed data (handle 0, negative values)

def get_data_statistics(data:np.array) -> Tuple[float, float]:
    mu  = X_train.mean(axis=0)
    std = X_train.mean(axis=0)
    return mu, std 

