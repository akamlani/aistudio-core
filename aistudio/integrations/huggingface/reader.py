import  numpy  as np
import  pandas as pd
import  itertools
import  torch
from    typing import List, Optional


import  datasets
from    datasets.dataset_dict import DatasetDict
from    datasets.arrow_dataset import Dataset
from    datasets.features import Sequence


def transform_targets(dataset_dict:DatasetDict, partition:str = 'train', feature_name: str = 'label') -> DatasetDict:
    # e.g., map (0,1) to text labels based on the feature names
    classes = [cls_name.replace("_", " ") for cls_name in dataset_dict[partition].features[feature_name].names]
    return dataset_dict.map(
        lambda x: {f"text_{feature_name}": [classes[target] for target in x[feature_name]]},
        batched=True,
        num_proc=4
    )

class HuggingFaceReader(object):

    def __init__(self, uri:str, name:Optional[str]=None, streaming:bool=False, **kwargs):
        """_summary_

        Args:
            uri (str): _description_
            name (Optional[str], optional): _description_. Defaults to None.
            streaming (bool, optional): _description_. Defaults to False.

        >>> pretrained_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        >>> records:List[dict] = .sample(pretrained_dataset, n=5)
        """
        self.dataset:DatasetDict = datasets.load_dataset(uri, streaming=streaming, **kwargs)
        self.num_partitions      = len(self.dataset)
        self.partitions          = list(self.dataset.keys())
        self.compute_dtype       = getattr(torch, "float16")

    def config_collator(self, tokenizer: transformers.PreTrainedTokenizerBase) -> None:
        self.tokenizer          = tokenizer
        self.data_collator      = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)

    def load(self, name:str, split:str='train', streaming:bool=False) -> pd.DataFrame:
        return datasets.load_dataset(name, split=split, streaming=streaming, **kwargs)

    def load_split(self, name:str, split:str='train', streaming:bool=False) -> Dataset:
        # e.g. dataset = datasets.load_dataset("lewtun/github-issues", split="train", streaming=True)
        #pe.g.,example = next(iter(dataset))
        return datasets.load_dataset(name, split=split, streaming=streaming, **kwargs)

    def partition(self, dataset:Dataset, test_size:float=0.2, **kwargs) -> DatasetDict:
        return dataset.train_test_split(test_size=test_size, **kwargs)

    def get_partition_info(self) -> pd.DataFrame:
        data = self.dataset
        return pd.DataFrame([
            {
                "partition": partition,
                "dtype":     type(data[partition][0]),
                "shape":     data[partition].shape,
                "num_rows":  data[partition].num_rows,
                "features":  list(data[partition].features.keys()),
                "columns":   data[partition].column_names,
                "example":   data[partition][0]
            }
            for partition in data.keys()
        ])

    def from_pandas(self, df:pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(df)

    def to_pandas(self, dataset:Dataset) -> pd.DataFrame:
        return dataset.to_pandas()

    def get_example(self, dataset:Dataset) -> dict:
        return dataset[0]

    def extract_labels(self, partition:str='train', feature_name:str='ner_tags') -> List[str]:
        feature:Sequence = self.dataset[partition].features[feature_name]
        labels:datasets.ClassLabel = feature.feature
        return labels.names

    def sample(self, dataset:Dataset, n:int=5) -> List[dict]:
        return list( itertools.islice(dataset, n) )

    def encode(self, dataset:Dataset, col:str) -> Dataset:
        return dataset.class_encode_column(col)

    def collate(self, tokenizer):
        return datasets.DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(self, examples, tokenizer, col):
        # each example is an element from the Dataset
        return tokenizer(examples[col], truncation=True)

    def rename_col(self, dataset:Dataset, old:str, new:str) -> Dataset:
        return dataset.rename_column(old, new)

    def remove_cols(self, dataset:Dataset, cols:List[str]) -> Dataset:
        return dataset.remove_columns(cols)
