import pandas as pd 

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
