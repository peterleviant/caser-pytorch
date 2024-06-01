import os
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Type
import polars as pl
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from caser_datasets.utils import apply_action_with_flag, get_data_dir, DatasetDescription

class SequentialRecommenderDataset(Dataset):
    _PREPROCESSED_FLAG: str = "__data_preprocessed"
    _ITEMS_FILE: str = "items.parquet"
    _USERS_FILE: str = "users.parquet"
    _DATA_FILE: str = "data.parquet"
    _METADATA_DIR: str = "metadata"

    items: pl.DataFrame
    users: pl.DataFrame
    data: pl.DataFrame
    _interactions: pl.DataFrame
    _future_items: pl.DataFrame
    _past_items: pl.DataFrame
    _item_history: torch.Tensor
    _item_label: torch.Tensor
    _user: torch.Tensor
    metadata: Dict[str, pl.DataFrame]

    def __init__(self, cold_start_count: int, dataset: DatasetDescription, base_dir: Optional[str]):
        self._cold_start_count: int = cold_start_count
        self._name: str = dataset.name
        self._data_dir = get_data_dir(dataset.name, base_dir)
        self._load_dataset(dataset)

    def _prepare_interactions(self, L: int, skip: int) -> None:
        self._interactions = self.data.lazy().rolling(index_column="index", period=f"{L+1+skip}i", by="user").agg([
            pl.col("item").list.to_struct(n_field_strategy='max_width',upper_bound=L+skip+1).alias("item"),
            pl.col("interation_number").list.get(L).alias("interation_number"),
            pl.col("interactions_count").last().alias("interactions_count")
        ]).unnest("item").select(
            ["index","user","interation_number","interactions_count"] + [f"field_{i}" for i in range(L+1+skip)]
        ).filter(
            pl.col(f"field_{L+skip}").is_not_null()
        ).rename({
            f"field_{i}":f"item_t_minus_{L-i}" for i in range(0, L)
        }).rename({f"field_{L + i}":f"item_t_plus_{i}" for i in range(skip+1)}).collect()
        print(f"Interactions set for {self._name}, L={L}, skip={skip}")


    def set_split(self, split: str, train_percentage: float, val_percentage: float, test_percentage: float) -> None:
        min_percentage = 0 if split == "train" else (train_percentage if split == "validation" else train_percentage + val_percentage)
        max_percentage = train_percentage if split == "train" else (train_percentage + val_percentage if split == "validation" else train_percentage + val_percentage + test_percentage)
        self._interactions = self._interactions.filter(
            (pl.col("interation_number") <= pl.col("interactions_count") * max_percentage) & \
            (pl.col("interation_number") > pl.col("interactions_count") * min_percentage)
        )
        if split == "validation" or split == "test":
            self._interactions = self._interactions.groupby("user").first().sort(by="user")
            user_with_test_interaction_number = self._interactions.select(["user","interaction_number"]).with_columns([
                pl.col("interaction_number").alias("test_interation_number")   
            ])
            data_with_test_indices = self.data.join(user_with_test_interaction_number, on="user")
            future_items = data_with_test_indices.filter(pl.col("interation_number") >= pl.col("test_interation_number")).select(['user','item']).distinct()
            past_items = data_with_test_indices.filter(pl.col("interation_number") < pl.col("test_interation_number")).select(['user','item']).distinct()
            future_items = future_items.group_by("user").agg([pl.col("item").list().alias("future_items")])
            past_items = past_items.group_by("user").agg([pl.col("item").list().alias("past_items")])
            self._future_items = user_with_test_interaction_number.select(["user"]).join(future_items, on="user", how="left").fill_none("future_items", [])
            self._past_items = user_with_test_interaction_number.select(["user"]).join(past_items, on="user", how="left").fill_none("past_items", [])
        else:
            self._past_items = self._future_items = len(self._interactions) * [[]]
        self._user = torch.LongTensor(self._interactions["user"])
        self._item_history = torch.LongTensor(
            self._interactions.select([c for c in self._interactions.columns if c.startswith("item_t_minus")]).to_numpy())
        self._item_label = torch.LongTensor(
            self._interactions.select([c for c in self._interactions.columns if c.startswith("item_t_plus")]).to_numpy())

    def set_L_and_skip(self, L: int, skip: int) -> None:
        self._interactions = self.data.lazy().select(["user","item","timestamp"]).sort(["user","timestamp"]).with_columns([
            pl.arange(start=0, end=len(self.data)).alias("index")
        ]).rolling(index_column="index", period=f"{L+1+skip}i", by="user").agg([pl.col("item")]).with_columns([
            pl.col("item").list.to_struct(n_field_strategy='max_width',upper_bound=L+skip+1)
        ]).unnest("item").select(
            ["index","user"] + [f"field_{i}" for i in range(L+1+skip)]
        ).filter(
            pl.col(f"field_{L+skip}").is_not_null()
        ).rename({
            f"field_{i}":f"item_t_minus_{L-i}" for i in range(0, L)
        }).rename({f"field_{L + i}":f"item_t_plus_{i}" for i in range(skip+1)}).collect()
        print(f"Interactions set for {self._name}, L={L}, skip={skip}")

    def _load_dataset(self, dataset: DatasetDescription) -> None:
        apply_action_with_flag(
            self._data_dir,
            dataset,
            self._preprocess_data,
            self._load_preprocessed_data,
            self._PREPROCESSED_FLAG
        )

    def _preprocess_data(self, *args) -> None:
        users, items, data, self.metadata = self._preprocess_data_inner()
        items, users, data = self._filter_cold_start(items, users, data)
        self.items, self.users, self.data = self._set_user_and_item_indices(items, users, data)
        self.data = self.data.sort(by=["user","timestamp"]).with_columns([
            (pl.col("user").cum_count().over("user") + 1).alias("interation_number"),
            pl.col("user").count().over("user").alias("interactions_count"),
            pl.arange(start=0, end=len(self.data)).alias("index")
        ])
        self._save_preprocessed_data()
        self._remove_raw_data()

    def _load_preprocessed_data(self, *args) -> None:
        self.items = self._load_dataframe(SequentialRecommenderDataset._ITEMS_FILE)
        self.users = self._load_dataframe(SequentialRecommenderDataset._USERS_FILE)
        self.data = self._load_dataframe(SequentialRecommenderDataset._DATA_FILE)
        metadata_dir = SequentialRecommenderDataset._METADATA_DIR
        self.metadata = {
            k.split(".parquet")[0]: self._load_dataframe(os.path.join(metadata_dir, k)) for k in 
            os.listdir(os.path.join(self._data_dir,metadata_dir)) if k.endswith(".parquet")
        }

    def _set_user_and_item_indices(self, items: pl.DataFrame, users: pl.DataFrame, data: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        users = users.with_columns([pl.arange(0, len(users)).alias("user_id")])
        items = items.with_columns([pl.arange(0, len(items)).alias("item_id")])
        data = data.join(users.select(["user","user_id"]), on="user").join(items.select(["item","item_id"]), on="item").drop(
            ["user","item"]).rename({"user_id":"user","item_id":"item"})
        users = users.rename({"user":"user_original","user_id":"user"})
        items = items.rename({"item":"item_original","item_id":"item"})
        return items, users, data

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
        users = users.join(data.select("user").unique(), on="user", how="inner")
        items = items.join(data.select("item").unique(), on="item", how="inner")

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
        os.makedirs(os.path.join(self._data_dir, metadata_dir), exist_ok=True)
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dataframe.write_parquet(file_path)

    def __len__(self) -> int:
        return len(self._user)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor,torch.LongTensor,torch.LongTensor]:
        return (self._user[idx], self._item_history[idx,:]), self._item_label[idx,:], self._past_items[idx,:], self._future_items[idx,:]

class SequentialRecommenderDataModule(pl.LightningDataModule): 
    def __init__(
            self,
            dataset_class: Type[SequentialRecommenderDataset],
            cold_start_count: int,
            dataset_description: DatasetDescription,
            base_dir: Optional[str],
            L: int,
            skip: int,
            train_percentage: float,
            val_percentage: float,
            test_percentage: float,
            batch_size: int,
            num_workers: int
        ):
        super().__init__()
        self.dataset_class = dataset_class
        self.cold_start_count = cold_start_count
        self.dataset_description = dataset_description
        self.base_dir = base_dir
        self.L = L
        self.skip = skip
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets: Dict[str, SequentialRecommenderDataset] = {}
        self.create_dataset_splits()

    def create_dataset_splits(self) -> None:
        self.datasets = {}
        dataset_splits = ["train","validation","test"]
        for split in dataset_splits:
            self.datasets[split] = self.dataset_class(self.cold_start_count, self.dataset_description, self.base_dir)
            self.datasets[split].set_L_and_skip(self.L, self.skip)

    def train_dataloader(self):
        return DataLoader(self["train"], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self["validation"], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self["test"], batch_size=self.batch_size, num_workers=self.num_workers)
