import pandas as pd 
import os
from   typing import Optional
from  ..core.io.filesystem import get_uri_properties, search_files_to_dataframe


class Catalog(object):
    def __init__(self, uri:str, ext:Optional[str]=None):
        self.df_catalog:pd.DataFrame = self.read_data_catalog(uri, ext)

    def read_data_catalog(self, uri:str, ext:Optional[str]=None) -> pd.DataFrame: 
        ext = f"*/*.{ext}" if ext else "*/*"
        return (
            search_files_to_dataframe(uri=str(uri), extension=ext).assign(
                db = lambda df_: df_.apply(lambda row: row['uri'].split(uri)[-1].rstrip(row['filename']).strip(os.sep), axis=1)
            )
        ).sort_values(
            by='db', ascending=True
        ).rename(
            columns={'name': 'table'}
        ).reset_index(drop=True)[['db', 'table', 'filename', 'uri']]
        # TODO: Drop *.yaml files (datasets.yaml)

    def read_catalog_metrics(self, df:pd.DataFrame) -> dict:
        return dict(
            num_files    = len(df['table']),
            num_dbs      = df['db'].nunique()
        )

    def read_datasource_properties(self, db_name:str, table_name:str) -> dict:
        # e.g., source = cat.read_datasource('imdb', 'directors')
        # e.g., df = pd.read_csv(source['uri])
        uri                 = self.df_catalog[self.df_catalog["table"] == table_name].uri.values[0]
        uri_properties:dict = get_uri_properties(uri)
        return uri_properties 
