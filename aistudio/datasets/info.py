import  pandas as pd 
import  numpy  as np 
import  scipy.stats as scs 
from    typing import List, Union, Optional
from    types import SimpleNamespace
from    collections import Counter 

class OutlierDetection(object):
    @classmethod 
    def std_is_outlier(cls, ds:pd.Series, k:int=2) -> dict: 
        stats:SimpleNamespace = SimpleNamespace(**InfoTabular.calc_col_stats(df_ticker_dt_flt[column_sel]) )
        is_outlier:dict = {
            f"gt_{k}*std_is_outlier" : (np.abs(ds - stats.mean) > k * stats.std).astype(int)
        }

    @classmethod 
    def iqr_is_outlier(cls, ds:pd.Series) -> dict:
        stats_iqr:SimpleNamespace = SimpleNamespace(**InfoTabular.calc_iqr_range(df_ticker_dt_flt[column_sel]) )
        is_outlier:dict = {
            "is_iqr_outlier" : ((ds < props["lower_bound"]) | (ds > props["upper_bound"])).astype(int)
        }    


class SchemaInfo(object):
    # numerical_cols   = lambda df_: df_.select_dtypes(include=['float64', 'int']).columns
    # categorical_cols = lambda df_: df_.select_dtypes(include=['object']).columns
    get_data_dtypes  = lambda df_: {col: str(df_[col].dtype) for col in df_.columns}

    @classmethod
    def filter_dataframe(cls, df: pd.DataFrame) -> dict:
        filter_type = ["number", "object", "datetime", "category", "bool"]
        return df.select_dtypes(include=filter_type)

    @classmethod
    def trsfrm_col_category_dtype(cls, df: pd.DataFrame, cols:List[str]) -> dict: 
        # https://github.com/anonymouskitler/kaggle/blob/master/Pytorch%20data%20loader%20%26%20category%20embeddings.ipynb
        for col in cols:
            df[col] = df[col].astype('category').cat.as_ordered()
        return df 

    @classmethod
    def get_schema_info(cls, df: pd.DataFrame) -> pd.DataFrame:
        # columns become index
        col_dtype_name = 'data_dtype'
        df_dtype_infer = (df.infer_objects().dtypes).to_frame(f'{col_dtype_name}_inferred')
        df_schema_info =  pd.DataFrame.from_dict(
            [cls.get_data_dtypes(df), cls.get_schema_dtype(df)],
        ).T.rename(columns={0:col_dtype_name, 1:f"logical_{col_dtype_name}"})
        return pd.concat([df_schema_info, df_dtype_infer], axis=1)[[
            col_dtype_name, f'{col_dtype_name}_inferred', f"logical_{col_dtype_name}"
        ]]

    @classmethod
    def get_schema_dtype(cls, df: pd.DataFrame) -> dict:
        properties = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                dtype_ = "numerical"
            elif pd.api.types.is_bool_dtype(dtype):
                dtype_ = "boolean"
            elif pd.api.types.is_object_dtype(dtype):
               if   cls.is_categorical(df, col): dtype_ =  "categorical"
               elif cls.is_datetime(df, col):    dtype_ =  "date"
               else: dtype_ = "string"
            # will be deprecated in future: pd.api.types.is_categorical_dtype(dtype)
            elif isinstance(dtype, pd.api.types.CategoricalDtype):
                dtype_ = "categorical"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                dtype_ = "date"
            else:
                dtype_ = "unknown"
            properties[col] = dtype_
        return properties

    @classmethod
    def is_categorical(cls, df: pd.DataFrame, col: str) -> bool:
        return  (
            # will be deprecated in future: pd.api.types.is_categorical_dtype(dtype)
            True if isinstance(df[col].dtype, pd.api.types.CategoricalDtype)
            else (
                True if df[col].nunique() / len(df[col]) < 0.5
                else False
            )
        )

    @classmethod
    def is_datetime(cls, df: pd.DataFrame, col: str) -> bool:
        if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            return True
        else:
            try:
                _ = pd.to_datetime(df[col], errors='raise', format='mixed')
                return True
            except ValueError:
                return False




class InfoTabular(object):
    @classmethod
    def describe(cls, df:pd.DataFrame, numerical:bool=False) -> pd.DataFrame:
        flt = [np.number] if numerical else [object, np.number, np.datetime64]
        return df.describe(include=flt)

    @classmethod 
    def calc_stats(cls, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({col: cls.calc_col_numerical_stats(df[col]) for col in df.columns}).T

    @classmethod
    def calc_row_stats(cls, df:pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{
            f"|{col}-mean|": np.abs(df[col] - df[col].mean()) for col in df.columns
        })

    @classmethod 
    def calc_col_numerical_stats(cls, ds:pd.Series) -> dict:
        """
        Kurtosis: quantify the shape of the distribution, determine the peakiness, heaviness of the distribution
        - measures the sharpness of the peak fo the frequency distribution curve , a measure of outliers present in the distribution
        - distribution with higher kurosis has a heavier tail 
        - + means pointy, - means flat 
        - excess kurtosis: determine by subtracting 3 from kurtosis.  makes the normal distribution kurtosis equal .
        - tall and thin (kurtosis > 3): near the mean or at the extremes 
        - flat distribution (kurtosis <3): moderately spread out
        - (kurtosis == 3): looks more close to normal distribution
        - high kurtosis is an indicator that data has heavy outliers
        - low  kurtosis is an indicator that the data has a lack of outliers

        Skewness: measure of asymmetry of a distribution
        - between (-0.5, 0.5): symmetrical.  for normaly distributed data, skewness should be about (0).
        - between (-1, -0.5) or (0.5, 1): data is slightly moderately skewed 
        - less than -1 or greater than 1: high skewed 
        - when negative, tail of distribution is longer toward left-side of curve 
        - when positive, tail of distribution is longer toward right-side of curve
        """
        dtype = ds.dtype 
        stats = dict()
        
        def _range(xs): 
            return (xs.max() - xs.min())

        if pd.api.types.is_numeric_dtype(dtype):
            stats = {
                fn.__name__.lstrip('_'):round(fn(ds), 3)
                for fn in [np.min, np.max, np.median, _range, np.mean, np.std, scs.kurtosis, scs.skew]
            }
            stats |= {fn.__name__:round( fn(ds,  ddof=1, nan_policy='omit'), 3) for fn in [scs.variation]}

        return stats

    @classmethod
    def calc_iqr_range(cls, ds:pd.Series) -> dict:
        q25, q75 = np.percentile(ds, [25 ,75])
        iqr      = q75-q25
        lower_bound, upper_bound = ( (q25 - (1.5*iqr)), (q75 + (1.5*iqr)) )
        return dict(iqr = iqr, q1 = q25, q3 = q75, lower_bound = lower_bound, upper_bound = upper_bound)

    @classmethod
    def get_samples(cls, df: pd.DataFrame, n:int=3, seed:int=42) -> pd.Series:
        return pd.Series({
            col: df[col].dropna().sample(n=min(n, df[col].dropna().count()), random_state=seed).tolist()
            for col in df.columns
        }, name="samples")

    @classmethod 
    def calc_col_categorical_stats(cls, ds:pd.Series, k:Optional[int]=None) -> dict:
        cnt       = Counter(ds)
        dat       = pd.Series(cnt).sort_values(ascending=False)

        total_sz  = sum(cnt.values())
        mc        = cnt.most_common(k) if k is not None else cnt.most_common()
        kv_cnt    = dict(mc)
        kv_pct    = {field_name:round(((count/total_sz)*100),3) for field_name, count in kv_cnt.items()} 

        cardinality  = len(cnt)
        categories_k = list(kv_cnt)
        top, bottom  = dat.index[0], dat.index[-1]  
        return (
            dict(cardinality=cardinality, top=top, bottom=bottom) 
            | {f'categories_top_{k}'         if k else 'categories': categories_k} 
            | {f'distribution_cnt_top_{k}'   if k else 'distribution_cnt': kv_cnt} 
            | {f'distribution_pct_top_{k}'   if k else 'distribution_pct': kv_pct}
        )

    @classmethod
    def calc_categorical_distribution(cls, df:pd.DataFrame, col:str) -> pd.Series:
        "calculates distribution of column"
        return df[col].value_counts().to_frame('count')

    @classmethod
    def calc_sparsity(cls, df:pd.DataFrame) -> pd.DataFrame:
        "calculates missing values"
        return (
            df.isnull().sum().to_frame('count').astype(int)
            .assign(
                pct = lambda df_: df_['count'].transform(lambda x: x/len(df)*100)
            ).sort_values(
                by=['count'], ascending=False
            ).round(3)
        )

class InfoText(object):
    @classmethod
    def calc_record_statistics(cls, df: pd.DataFrame, col: str='text') -> pd.DataFrame:
        # can then compute global len |D| (min, max, std, avg) statistics across all records
        return df.assign(
            char_sz   =  lambda df_: df_[col].apply(len),
            word_sz   =  lambda df_: df_["words"].apply(len),
            token_sz  =  lambda df_: df_["tokens"].apply(len),
        )

    @classmethod 
    def calc_frame_statistics(cls, df: pd.DataFrame) -> pd.DataFrame:
        cols_numeric = df.select_dtypes(include=['number']).columns
        return df[cols_numeric].agg(['sum', 'min', 'max', 'std', 'mean']).round(3)


class InfoDateTime(object):
    # record operations
    @classmethod
    def calc_dt_stats(cls, data: Union[pd.DataFrame, pd.Series], col: Optional[str] = 'date'):
        if isinstance(data, pd.DataFrame):
            data_ts = data.index if data.index.name is not None and col in data.index.name else data[col]
        elif isinstance(data, pd.Series): 
            data_ts = data.index 

        return dict(
            min_date = min(data_ts),
            max_date = max(data_ts)
        )

    @classmethod
    def calc_dt_timespan(cls, start_time: np.datetime64, end_time: np.datetime64) -> dict:
        # division for years is to account leap years
        calc_dt_duration          = lambda start, end: abs(end - start)
        #trsfrm_timedelta_to_years = lambda td: (td / np.timedelta64(1, 'Y'))
        trsfrm_timedelta_to_years = lambda td: (td / 365.25)
        trsfrm_timedelta_to_qtrs  = lambda td: (td / (np.timedelta64(1, 'D')) / (30 * 3))

        duration = calc_dt_duration(start_time, end_time)
        return dict(
            days     = round(duration.days, 2),
            quarters = round(trsfrm_timedelta_to_qtrs(duration), 2),
            years    = round(trsfrm_timedelta_to_years(duration.days), 2)
        )



