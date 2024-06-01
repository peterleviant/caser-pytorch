from typing import Any, Callable, Optional, Tuple, Dict
from caser_datasets.huggingface import HuggingFaceDataset
from caser_datasets.sequential_recommender import SequentialRecommenderDataset
from caser_datasets.utils import DatasetDescription
import os
import polars as pl


class AmazonDataset(HuggingFaceDataset, SequentialRecommenderDataset):
    DATASET_PATH = "McAuley-Lab/Amazon-Reviews-2023"

    def _preprocess_data_inner(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, pl.DataFrame]]:
        """Main method to handle data preprocessing and file output."""
        data = pl.DataFrame(self.hf_dataset.data.table).rename({"parent_asin":"item_id"})
        data = data.cast({"timestamp":pl.Int64, "rating":pl.Float64})
        items = data.select(["item_id"]).unique()
        items = items.with_columns([pl.arange(0,items.shape[0]).alias("item")])
        users = data.select(["user_id"]).unique()
        users = users.with_columns([pl.arange(0,users.shape[0]).alias("user")])
        data = data.join(items, on="item_id")
        data = data.join(users, on="user_id")
        data = data.select(["user", "item", "rating", "timestamp"])
        return users, items, data, {}
    
    def _remove_raw_data(self) -> None:
        pass

    def __init__(
            self,
            hf_split: str = "full",
            dataset_name: Optional[str] = None,
            cold_start_count: int = 5,
            base_dir: Optional[str] = None,
            **kwargs
        ):
        HuggingFaceDataset.__init__(
            self, self.DATASET_PATH, hf_split, f"0core_rating_only_{dataset_name}", trust_remote_code = True, **kwargs)
        SequentialRecommenderDataset.__init__(self, cold_start_count, DatasetDescription(name=f"amazon/{dataset_name}"), base_dir=base_dir,
         **kwargs)
