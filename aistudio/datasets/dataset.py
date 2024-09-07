import  numpy as np
import  pandas as pd
from    dataclasses import dataclass
from    typing import TypeVar, Generic, List, Optional, Tuple 

from    .info import InfoTabular, InfoDateTime, SchemaInfo
from    ..core.transform import trsfrm_dt_features_tod, trsfrm_timestamp_to_dt


T_co    = TypeVar('T_co', covariant=True)
T       = TypeVar('T')


def text_wrap(text:str, width:int=120):
    """Wraps the given text to the specified width.

    Args:
        text (str): The text to wrap.
        width (int): The width to wrap the text to.

    Returns:
        str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])



@dataclass 
class DatasetStatistics(object):
    stats:pd.DataFrame 
    cat_stats:pd.DataFrame 
    cat_dist:pd.DataFrame

class DatasetT(Generic[T_co]):
    def __init__(self, **kwargs):
        self.data = None

    def __len__(self) -> int:
        try:
            return len(self.data)
        except e:
            raise NotImplementedError


class DatasetTabular(DatasetT):
    def __init__(self,
        df:pd.DataFrame,
        chunksize:int=32,
        *args,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data       = df
        self.indices    = df.index
        self.shape      = df.shape
        self.num_obs, self.num_features = df.shape
        self.num_chunks = len(df) // chunksize
        # get schema information
        self.get_schema()
        # self.data = SchemaInfo.trsfrm_col_category_dtype(self.data, self.cat_cols)
        # get statistics 
        self.get_properties()
        self.get_statistics()


    def __len__(self) -> int:
        return len(data)

    def __getitem__(self, idx:int) -> pd.Series:
        return self.data.iloc[idx]

    def __repr__(self):
        return f"Class: {self.__class__.__name__} | Shape: {self.data.shape}"

    def get_schema(self) -> pd.DataFrame:
        # compute Statistics and Missing Values
        self.df_schema        = SchemaInfo.get_schema_info(self.data)
        self.numerical_cols   = self.df_schema[self.df_schema['schema_dtype'] == 'numerical'].index
        self.cat_cols         = self.df_schema[self.df_schema['schema_dtype'] == 'categorical'].index
        self.text_cols        = self.df_schema[self.df_schema['schema_dtype'] == 'string'].index
        return self.df_schema

    def describe(self) -> pd.DataFrame:
        return self.data[self.numerical_cols].describe(include=['number']).T 

    def get_properties(self) -> pd.DataFrame:
        self.df_sparsity     = InfoTabular.calc_sparsity(self.data[self.numerical_cols])
        return self.df_sparsity 

    def get_statistics(self) -> None:
        if self.numerical_cols is not None and len(self.numerical_cols) > 0:
            self.numerical_stats =  InfoTabular.calc_stats(self.data[self.numerical_cols])
        if self.cat_cols is not None and len(self.cat_cols) > 0:
            self.cat_stats = self.data[list(self.cat_cols)].describe(include=['object'])
            self.cat_dist  = InfoTabular.calc_distribution(self.data, col=self.cat_cols)

    def get_timespan(self, col:str='date') -> dict:
        dt_props:dict        = InfoDateTime.calc_dt_stats(self.data, col=col)
        dt_span:dict         = InfoDateTime.calc_dt_timespan(dt_props['min_date'], dt_props['max_date'])
        return (dt_props | dt_span)


class DatasetTimeSeries(DatasetTabular):
    def __init__(self,
        df:pd.DataFrame,
        chunksize:int=32,
        *args,
        **kwargs
    ):
        super().__init__(df, chunksize, args, kwargs)



class DatasetClassifier(DatasetTabular):
    def __init__(self,
        df:pd.DataFrame,
        target_col:str, 
        chunksize:int=32,
        *args,
        **kwargs
    ):
        super().__init__(df, chunksize, args, kwargs)
        # if classification
        if target_col and isinstance(self.data[target_col], object):
            self.target_col  = target_col
            self.targets     = self.data[target_col].unique().tolist()
            self.num_classes = self.data[target_col].nunique()


    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.array]: 
        X = df.drop(self.target_col, axis=1)
        y = df[target_col]
        return X[idx], y[idx]

    def assignment(self, df:pd.DataFrame) -> Tuple[dict, dict]:
        # used for contrastive learning (contrastive loss)
        labels_positive = dict()
        labels_negative = dict()
        for target in self.targets: 
            labels_positive = df[df[target_col] == target].to_numpy()
        for target in self.targets: 
            labels_negative = df[df[target_col] != target].to_numpy()

        return labels_positive, labels_negative