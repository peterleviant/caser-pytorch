import os
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict
import polars as pl

from caser_datasets.utils import apply_action_with_flag, get_data_dir

class SequentialRecommenderDataset(Dataset):
    _PREPROCESSED_FLAG: str = "__data_preprocessed"
    _ITEMS_FILE: str = "items.parquet"
    _USERS_FILE: str = "users.parquet"
    _DATA_FILE: str = "data.parquet"
    _METADATA_DIR: str = "metadata"

    items: pl.DataFrame
    users: pl.DataFrame
    data: pl.DataFrame
    metadata: Dict[str, pl.DataFrame]

    def __init__(self, cold_start_count: int, name: str, base_dir: Optional[str]):
        self._cold_start_count: int = cold_start_count
        self._name: str = name
        self._data_dir = get_data_dir(name, base_dir)
        self._load_dataset()

    def _load_dataset(self) -> None:
        apply_action_with_flag(
            self._data_dir,
            self._name,
            self._preprocess_data,
            self._load_preprocessed_data,
            self._PREPROCESSED_FLAG
        )

    def _preprocess_data(self) -> None:
        items, users, data, self.metadata = self._preprocess_data_inner()
        self.items, self.users, self.data = self._filter_cold_start(items, users, data)
        self._save_preprocessed_data()
        self._remove_raw_data()

    def _load_preprocessed_data(self) -> None:
        self.items = self._load_dataframe(SequentialRecommenderDataset._ITEMS_FILE)
        self.users = self._load_dataframe(SequentialRecommenderDataset._USERS_FILE)
        self.data = self._load_dataframe(SequentialRecommenderDataset._DATA_FILE)
        metadata_dir = SequentialRecommenderDataset._METADATA_DIR
        self.metadata = {
            k.split(".parquet")[0]: self._load_dataframe(os.path.join(metadata_dir, k)) for k in os.listdir(metadata_dir) if k.endswith(".parquet")
        }

    def _filter_cold_start(self, items: pl.DataFrame, users: pl.DataFrame, data: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        k = self._cold_start_count
        while True:
            # Compute degree counts using groupby and aggregation
            old_len = len(data)
            data = data.filter(
                (pl.col("user").count().over("user").alias("user_count") >= k) &
                (pl.col("item").count().over("item").alias("item_count") >= k)
            )
            if len(data) == old_len:
                break

        users = users.join(data.select("user"), on="user", how="inner")
        items = items.join(data.select("item"), on="item", how="inner")

        return items, users, data

    def _preprocess_data_inner(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        raise NotImplementedError()
    
    def _remove_raw_data(self):
        raise NotImplementedError()

    def _save_preprocessed_data(self) -> None:
        self._write_dataframe(SequentialRecommenderDataset._ITEMS_FILE, self.items)
        self._write_dataframe(SequentialRecommenderDataset._USERS_FILE, self.users)
        self._write_dataframe(SequentialRecommenderDataset._DATA_FILE, self.data)
        
        metadata_dir = SequentialRecommenderDataset._METADATA_DIR
        for key, value in self.metadata.items():
            file_name = f"{key}.parquet"
            self._write_dataframe(os.path.join(metadata_dir, file_name), value)

    def _load_dataframe(self, file_name: str) -> Optional[pl.DataFrame]:
        file_path = os.path.join(self._data_dir, file_name)
        if os.path.isfile(file_path):
            return pl.read_parquet(file_path)
        else:
            print(f"File {file_path} does not exist.")
            return None

    def _write_dataframe(self, file_name: str, dataframe: pl.DataFrame) -> None:
        file_path = os.path.join(self._data_dir, file_name)
        dataframe.write_parquet(file_path)
