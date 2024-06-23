import  numpy as np
import  pandas as pd
from    typing import TypeVar, Generic, List, Optional
from    .info import InfoTabular

T_co    = TypeVar('T_co', covariant=True)
T       = TypeVar('T')

class DatasetT(Generic[T_co]):
    def __init__(self, **kwargs):
        self.data = None

    def __len__(self) -> int:
        try:
            return len(self.data)
        except e:
            raise NotImplementedError
