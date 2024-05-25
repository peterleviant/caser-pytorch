from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import datasets as hf_datasets

class HuggingFaceDataset(Dataset):
    hf_dataset: hf_datasets.Dataset

    def __init__(
        self,
        dataset_path: str,
        split: str,
        dataset_name: Optional[str] = None,
        **kwargs
    ):
        self.hf_dataset = hf_datasets.load_dataset(dataset_path, dataset_name, split=split, **kwargs)
        self.hf_dataset.set_format(type="torch", columns=None)
