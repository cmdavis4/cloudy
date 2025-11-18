from typing import Any
import pandas as pd
import xarray as xr
from dataclasses import dataclass
import datetime as dt
from pathlib import Path


class ValidatedDict(dict):

    def __getattribute__(self, name: str) -> Any:
        return super(ValidatedDict, self).__getattribute__(name).validate()


class DataSet:

    def __init__(self, data) -> None:
        self.data = data

    def validate(self):
        pass


class DataWrapper:

    def __init__(self) -> None:
        self._data = ValidatedDict()

    def add_data(self, **data_kwargs):
        for name, data in data_kwargs.items():
            self._data[name] = DataSet(data).validate()

    @property
    def data(self):
        return self._data
