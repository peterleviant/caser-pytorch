from enum import Enum
import os
from typing import Dict, Optional, Tuple
from caser_datasets.sequential_recommender import SequentialRecommenderDataset
from caser_datasets.url_zipped import URLZippedDataset
from typing import Optional
import polars as pl


class GowallaDataset(URLZippedDataset, SequentialRecommenderDataset):
    class Datasets(Enum):
        GOWALLA_CHECK_IN = URLZippedDataset.DatasetDescription(
            url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
            name = "GowallaCheckIns"
        )

    _DATA_RAW_FILE: str = 'data_format1/user_log_format1.csv'

    def _preprocess_data_inner(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, pl.DataFrame]]:
        """Main method to handle data preprocessing and file output."""
        file_path = os.path.join(self._data_dir, self._DATA_RAW_FILE)
        schema = {"user": pl.Int32, "check_in_time": pl.Utf8, "latitude": pl.Float32, "longitude": pl.Float32, "location_id": pl.Int64}
        data = pl.read_csv(file_path, schema=schema, separator="\t").rename({"check_in_time":"timestamp","location_id":"item"})
        data = data.with_columns(pl.col("timestamp").str.to_datetime('%Y-%m-%dT%H:%M:%SZ', strict=False).dt.timestamp().alias("timestamp"))
        items = data.select("item").unique()
        users = data.select("user").unique()
        return users, items, data, {}
    
    def _remove_raw_data(self) -> None:
        file_path = os.path.join(self._data_dir, self._DATA_RAW_FILE)
        os.remove(file_path)

    def __init__(
            self,
            dataset_to_use: URLZippedDataset.DatasetDescription,
            cold_start_count: int = 5,
            base_dir: Optional[str] = None
    ):
        URLZippedDataset.__init__(self, dataset_to_use, base_dir=base_dir)
        SequentialRecommenderDataset.__init__(self, cold_start_count, dataset_to_use.name, base_dir=base_dir)
