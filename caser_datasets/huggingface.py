from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import datasets as hf_datasets

class HuggingFaceDataset(Dataset):
    _dataset: hf_datasets.Dataset
    _transform: Optional[Callable[[Any], Any]]

    def __init__(
        self,
        dataset_path: str,
        split: str,
        dataset_name: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        **kwargs
    ):
        self._dataset = hf_datasets.load_dataset(dataset_path, dataset_name, split=split, **kwargs)
        self._dataset.set_format(type="torch", columns=None)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        item = self._dataset[idx]

        # Apply the transform to the data if applicable
        if self._transform:
            data = self._transform(item)

        return data
