import calendar
import pandas as pd 
from   typing import List, Tuple  
from   datetime import datetime

# datetime transform constants
SECONDS     = 1000
MINUTES     = 60  * SECONDS
HOURS       = 60  * MINUTES
DAYS        = 24  * HOURS
WEEKS       = 7   * DAYS
MONTHS      = 30  * DAYS
YEARS       = 365 * DAYS
QUARTERS    = 3   * MONTHS

get_dt_format           = lambda: 'YYYY/MM/DD HH:MM:SS'
get_dt_periods          = lambda: {'M':1, 'Q':4, 'A':1}
get_dt_today            = lambda: datetime.now().strftime('%m-%d-%Y')
trsfrm_dt_to_str        = lambda x, sep: x.strftime(f"%Y{sep}%m{sep}%d %H:%M:%S")
trsfrm_col_to_date      = lambda df, col: pd.to_datetime(df[col], errors='raise')
trsfrm_timestamp_to_dt  = lambda xs: pd.Timestamp.to_pydatetime(xs).date() if isinstance(xs, pd.Timestamp) else xs


def get_dt_abbr() -> Tuple[dict, dict]:
    """
    Returns a dict of datetime abbreviations.
    """
    monthly_abbr:dict = { k: v for k, v in enumerate(calendar.month_abbr) if k }
    qrt_abbr:dict     = { i:f'Q{i}' for i in range( 1, 5 ) } 
    return monthly_abbr, qrt_abbr


def trsfrm_dt_features_tod(df:pd.DataFrame, col:str = 'date') -> pd.DataFrame:
    """transform dataframe with datetime fields

    Args:
        df (pd.DataFrame): input dataframe
        col (str, optional): datetime column to use. Defaults to 'Date'.

    Returns:
        pd.DataFrame: transformed dataframe with extracted date and time columns
    """
    df_dt = df.copy()
    if col not in df_dt.columns:
        df_dt[col]  = df_dt.index 

    return df_dt.assign(
        day     = lambda df_: df_[col].dt.day,
        month   = lambda df_: df_[col].dt.month,
        quarter = lambda df_: df_[col].dt.quarter,
        year    = lambda df_: df_[col].dt.year
    )

def trsfrm_frame_filter_dates(df:pd.DataFrame, col:str, start_date:str, end_date:str):
    return df[(df[col] >= start_date) & (df[col] <= end_date)]


def trsfrm_create_date(df:pd.DataFrame, cols:List[str], format="%Y/%d/%m") -> pd.DataFrame:
    """compute the date column as 'date' from existing columns
    
    Args:   
        df: dataframe to select columns from 
        cols: list of columns to select and transform a 'date' to 
        format: format of the date column

    Returns:
        dataframe with a 'date' column

    Example:
    >>> cols  = ['month', 'day', 'year']
    >>> df_dt = trsfrm_date(df_dt, cols)
    """
    df['date'] = pd.to_datetime(df[cols], format=format)
    return df
