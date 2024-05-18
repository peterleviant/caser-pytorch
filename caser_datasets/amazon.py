from typing import Any, Callable, Optional
from caser_datasets.huggingface import HuggingFaceDataset
from caser_datasets.sequential_recommender import SequentialRecommenderDataset


class AmazonDataset(HuggingFaceDataset, SequentialRecommenderDataset):
    DATASET_PATH = "McAuley-Lab/Amazon-Reviews-2023"

    def __init__(
            self,
            split: str = "full",
            dataset_name: Optional[str] = None,
            cold_start_count: int = 5,
            base_dir: Optional[str] = None,
            **kwargs
        ):
        HuggingFaceDataset.__init__(self.DATASET_PATH, split, dataset_name, trust_remote_code = True, **kwargs)
        SequentialRecommenderDataset.__init__(self, cold_start_count, f"amazon/{dataset_name}", base_dir=base_dir)
