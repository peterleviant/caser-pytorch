from torch.utils.data import Dataset
from typing import Optional, NamedTuple
import zipfile
import gzip
import gdown
import os
import shutil

import urllib.request

from caser_datasets.utils import apply_action_with_flag, get_data_dir, DatasetDescription

class URLZippedDataset(Dataset):
    _DOWNLOADED_FLAG: str = "__data_downloaded"

    def __init__(
            self,
            dataset_description: DatasetDescription,
            base_dir: Optional[str] = None,
        ) -> None:
        data_dir = get_data_dir(dataset_description.name, base_dir)
        none_action = lambda dataset_name, data_dir: None
        apply_action_with_flag(data_dir, dataset_description, self._download_data, none_action, self._DOWNLOADED_FLAG)

    def _unzip_gz(self, file_path: str) -> None:
        """
        Unzips a .gz file

        Args:
            file_path: The path to the .gz file.
        """
        with gzip.open(file_path, 'rb') as f_in, open(os.path.splitext(file_path)[0], 'wb') as f_out:
            f_out.write(f_in.read())

    def _unzip_zip(self, file_path: str, data_dir: str) -> None:
        """
        Unzips a .zip file

        Args:
            file_path: The path to the .gz file.
            data_dir: The directory to extract the file to.
        """
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    def _unzip_file(self, filename: str, data_dir: str) -> None:
        if filename.endswith(".zip"):
            self._unzip_zip(filename, data_dir)
        elif filename.endswith(".gz"):
            self._unzip_gz(filename)
        else:
            raise ValueError(f"Downloaded archive {filename} of type {filename.split('.')[-1]} is not supported")
        os.remove(filename)

    @staticmethod
    def _download_from_google_drive(gdrive_url: str, target_directory: str) -> str:
        # Download the file with gdown to the current directory with the original name
        downloaded_file_path: str = gdown.download(gdrive_url, output=None, quiet=False)
        # Ensure the target directory exists
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Create the final path for the file by joining the target directory with the basename of the downloaded file
        final_file_path: str = os.path.join(target_directory, os.path.basename(downloaded_file_path))

        # Move the downloaded file to the target directory
        shutil.move(downloaded_file_path, final_file_path)
        return final_file_path

    def _download_file(self, url: str, data_dir: str) -> str:
        if urllib.parse.urlparse(url).netloc == "drive.google.com":
            return self._download_from_google_drive(url, data_dir)
        else:
            filename: str = os.path.join(data_dir, os.path.basename(url))
            urllib.request.urlretrieve(url, filename)
            return filename

    def _download_data(self, dataset_description: DatasetDescription, data_dir: str) -> None:
        """
        Downloads an archive file from a url into a given directory, unzips it there into a new directory if does not
        already exists and then deletes the original file.

        Args:
            dataset_description: The description of the dataset to download.
            data_dir: The directory to download and extract the data to.
        """
        if os.path.isdir(data_dir):
            print(f"Data for dataset {dataset_description.name} already downloaded to {data_dir}")
            return
        os.makedirs(data_dir)

        # Download the zip file.
        try:
            filename: str = self._download_file(dataset_description.url, data_dir)
            self._unzip_file(filename, data_dir)
            print(f"Downloaded file from {dataset_description.url}")
        except:
            try:
                os.remove(filename)
                filename: str = self._download_file(dataset_description.mirror_url, data_dir)
                self._unzip_file(filename, data_dir)
                print(f"Downloaded file from mirror {dataset_description.mirror_url}")
            except Exception as e:
                raise RuntimeError("Failed to download dataset") from e
        print(f"Data for dataset {dataset_description.name} extracted into {data_dir}")
