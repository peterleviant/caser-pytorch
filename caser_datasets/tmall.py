from enum import Enum
import os
import shutil
from typing import Dict, Optional, Tuple
import polars as pl
from caser_datasets.url_zipped import URLZippedDataset
from caser_datasets.sequential_recommender import SequentialRecommenderDataset
from caser_datasets.utils import DatasetDescription

class TmallDataset(URLZippedDataset, SequentialRecommenderDataset):
    class Datasets(Enum):
        # TODO add other datasets
        TMALL = DatasetDescription(
            url="https://tianchi.aliyun.com/dataset/42?t=1715375717756",
            name="TmallDataFormat1",
            mirror_url="https://drive.google.com/uc?id=1SzgZLZRhMfT4mU5lxdNy8kgS4GqNFLwL"
        )

    _USER_RAW_FILE: str = 'data_format1/user_info_format1.csv'
    _DATA_RAW_FILE: str = 'data_format1/user_log_format1.csv'

    def _preprocess_data_inner(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, pl.DataFrame]]:
        """Main method to handle data preprocessing and file output."""
        user = pl.read_csv(os.path.join(self._data_dir, self._USER_RAW_FILE)).rename({"user_id": "user"})
        data = pl.read_csv(os.path.join(self._data_dir, self._DATA_RAW_FILE)).rename({"user_id": "user", "item_id": "item", "time_stamp": "timestamp"})
        items = data.select("item").unique()
        return user, items, data, {}
    
    def _remove_raw_data(self) -> None:
        shutil.rmtree(os.path.join(self._data_dir, "data_format1"), ignore_errors=True)

    def __init__(
            self,
            dataset_to_use: DatasetDescription,
            cold_start_count: int = 5,
            base_dir: Optional[str] = None, **kwargs
    ):
        URLZippedDataset.__init__(self, dataset_to_use, base_dir=base_dir)
        SequentialRecommenderDataset.__init__(self, cold_start_count, dataset_to_use, base_dir=base_dir, **kwargs)
