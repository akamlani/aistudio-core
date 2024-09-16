import numpy as np 
import pandas as pd 
from   typing import List, Dict, Any, Tuple, Union, Optional

from    ..core.io.writer_xls import ExcelFileWriter
from    .dataset    import DatasetTabular
from    .info       import InfoDateTime, InfoTabular, SchemaInfo


class DatasetReporter(object):
    """
    Examples:
    dt_stamp   = dataset.get_timespan('purchase_date')['min_date']
    filename   = DatasetReporter.autogen_filename( dataset_name='gna_appeals', timestamp=dt_stamp)
    >>> rp_writer  = DatasetReporter(dataset)
    >>> rp_writer.write_report(filename=filename, dt_col='purchase_date', k=4)
    """
    def __init__(self, dataset:DatasetTabular):
        self.writer = ExcelFileWriter() 
        self.dps    = dataset 

    def __repr__(self):
        return f"{self.dps}"

    @classmethod 
    def autogen_filename(cls, dataset_name:str, timestamp:pd.Timestamp) -> str:
        month_year = f'{timestamp.month:02d}{timestamp.year}'
        filename   = f'dataset_{dataset_name}_report_{month_year}.xlsx'
        return filename

    def get_completeness(self) -> pd.DataFrame:
        "build the schema and completenss in terms of sparsity"
        return pd.merge(
            self.dps.schema[['data_dtype','logical_data_dtype']],
            self.dps.sparsity.rename(columns={'count':'missing_count', 'pct':'missing_pct'}),
            left_index=True,
            right_index=True
        ).rename_axis('field')

    def get_global_stats(self, dt_col:Optional[pd.Timestamp]) -> pd.DataFrame:
        # add global stats 
        stats = dict(
            num_obs        = self.dps.num_obs,
            num_features   = self.dps.num_features,
            num_duplicated = self.dps.data.duplicated().sum()
        )
        # add memory stats
        stats.update(self.dps.get_memory_stats().sum().round(2).to_dict())
        # add datetime span stats
        if dt_col is not None:
            # time frame 
            dt_span:pd.DataFrame = pd.DataFrame(
                [self.dps.get_timespan(col) for col in self.dps.date_cols], 
                index=self.dps.date_cols
            ).rename_axis('field')
            stats.update( dict(dt_duration=dt_span.loc[dt_col, ['days','quarters','years']].to_dict() ) )
        # normalize and transpose column hierarchy 
        return (
            pd.json_normalize( stats )
            .set_axis(['stats'], axis=0)
        ).T.rename_axis('field')

    def write_report(self, filename:str, dt_col:Optional[pd.Timestamp], k:Optional[int]=10) -> dict:
        df_schema:pd.DataFrame            = self.dps.schema.astype(str).rename_axis('field')
        df_dataset_stats:pd.DataFrame     = self.get_global_stats(dt_col)
        df_completeness:pd.DataFrame      = self.get_completeness()
        df_categorical_stats:pd.DataFrame = self.dps.get_categorical_statistics(k)
        df_numerical_stats:pd.DataFrame   = self.dps.numerical_stats 

        mapping = {
            'dataset.schema'     : df_schema, 
            'stats.global'       : df_dataset_stats, 
            'stats.completeness' : df_completeness, 
            'stats.categorical'  : df_categorical_stats, 
            'stats.numerical'    : df_numerical_stats  
        }

        for sheet_name, frame in mapping.items():
            self.writer.write_excel(frame, filename, sheet_name, with_index=True)

        return mapping 

    def wrap_cell_content(self,value):
        if isinstance(value, list):
            return "\n".join(map(str, value))  # Join list items with a newline
        elif isinstance(value, dict):
            return "\n".join([f"{key}: {val}" for key, val in value.items()])  # Join dict items with a newline
        elif isinstance(value, str):
            return str(value)  # Return the value as is if it's not a list or dict
        else:
            return value 
