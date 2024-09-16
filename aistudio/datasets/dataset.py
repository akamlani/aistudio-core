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
        self.chunksize  = chunksize
        self.num_chunks = len(df) // chunksize
        self.num_obs, self.num_features = df.shape

        # get schema information
        self.get_schema()
        # self.data = SchemaInfo.trsfrm_col_category_dtype(self.data, self.cat_cols)

        # get statistics 
        self.get_properties()
        self.get_numerical_statistics()
        self.get_categorical_statistics()

    def __len__(self) -> int:
        return len(data)

    def __getitem__(self, idx:int) -> pd.Series:
        return self.data.iloc[idx]

    def __repr__(self):
        return (
            f"Class: {self.__class__.__name__} | Shape: {self.data.shape} | "
            f"Num Chunks: {self.num_chunks} | ChunkSize: {self.chunksize}"
        )

    def get_memory_stats(self, unit_mb:bool=False) -> pd.DataFrame:
        df = self.data
        # Memory usage of each column
        memory_usage    = df.memory_usage(deep=True)  
        # Total memory usage (in KB)
        unit_str        = 'MB' if unit_mb else 'KB'
        unit_trsfrm     = (1024**2) if unit_mb else (1024)
        total_memory    = memory_usage.sum()/unit_trsfrm
        memory_by_dtype = memory_usage.groupby(df.dtypes).sum() / unit_trsfrm
        return memory_by_dtype.rename_axis('dtype').to_frame(f'memory_{unit_str}')

    def get_schema(self) -> pd.DataFrame:
        # compute Statistics and Missing Values
        self.schema:pd.DataFrame = SchemaInfo.get_schema_info(self.data)
        self.numerical_cols   = self.schema[self.schema['logical_data_dtype'] == 'numerical'].index
        self.cat_cols         = self.schema[self.schema['logical_data_dtype'] == 'categorical'].index
        self.text_cols        = self.schema[self.schema['logical_data_dtype'] == 'string'].index
        self.date_cols        = self.schema[self.schema['logical_data_dtype'] == 'date'].index
        return self.schema

    def describe(self) -> pd.DataFrame:
        return self.data[self.numerical_cols].describe(include=['number']).T 

    def get_properties(self) -> pd.DataFrame:
        self.sparsity     = InfoTabular.calc_sparsity(self.data)
        return self.sparsity 

    def get_numerical_statistics(self) -> None:
        if self.numerical_cols is not None and len(self.numerical_cols) > 0:
            self.numerical_stats =  InfoTabular.calc_stats(self.data[self.numerical_cols])

    def get_categorical_statistics(self, k:Optional[int]=None) -> None:
        if self.cat_cols is not None and len(self.cat_cols) > 0:
            #self.cat_dist  = InfoTabular.calc_distribution(self.data, col=self.cat_cols)
            self.cat_basic_stats    = self.data[list(self.cat_cols)].describe(include=['object'])
            self.cat_advanced_stats =  pd.DataFrame(
                [InfoTabular.calc_col_categorical_stats(self.data[col], k) for col in self.cat_cols], 
                index=[self.cat_cols]
            ).rename_axis('field')
            return self.cat_advanced_stats

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