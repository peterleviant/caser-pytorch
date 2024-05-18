from enum import Enum
from caser_datasets.sequential_recommender import SequentialRecommenderDataset
from caser_datasets.url_zipped import URLZippedDataset
import os
import subprocess
import chardet
import shutil
from typing import Optional
import polars as pl


class MovieLensDataset(URLZippedDataset, SequentialRecommenderDataset):
    class Datasets(Enum):
        #TODO add other datasets
        MOVIE_LENS_1M = URLZippedDataset.DatasetDescription(
            url= "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            name  = "MovieLens1M"
        )

    _DATA_RAW_FILE: str = "ml-1m/ratings.dat"
    _ITEMS_RAW_FILE: str = "ml-1m/movies.dat"
    _USERS_RAW_FILE: str = "ml-1m/users.dat"

    def _preprocess_data(self) -> None:
        """Main method to handle data preprocessing and file output."""
        delimiters = []
        for filename in [self._DATA_RAW_FILE, self._ITEMS_RAW_FILE, self._USERS_RAW_FILE]:
            self._convert_file_to_utf8(filename)
            delimiters.append(self._set_valid_delimiter(filename)) # delimiter is :: which is not supported by polars/pyarrow
        
        data, items, users = self._read_ratings_data(delimiters[0]), self._read_movies_data(delimiters[1]), self._read_users_data(delimiters[2])
        return users, items, data, {}

    def _convert_file_to_utf8(self, filename: str) -> None:
        file_encoding = self._detect_encoding(filename)
        if file_encoding in ['UTF-8', 'ascii']:
            return
        file_path = os.path.join(self._data_dir, filename)
        temp_file_path = file_path + ".tmp"
        command = ['iconv', '-f', file_encoding, '-t', 'UTF-8', file_path]
        try:
            with open(temp_file_path, 'w') as temp_file:
                subprocess.run(command, stdout=temp_file, check=True)

            # Replace the original file with the converted temporary file
            os.replace(temp_file_path, file_path)
            print(f"Conversion successful. File {file_path} has been updated to UTF-8.")

        except subprocess.CalledProcessError as e:
            print(f"Conversion failed of file {file_path}:", e)
            # Clean up temporary file if conversion fails
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


    def _read_ratings_data(self, delimiter: str) -> pl.DataFrame:
        file_path = os.path.join(self._data_dir, self._DATA_RAW_FILE)
        schema = {"user_id": pl.Int32, "movie_id": pl.Int32, "rating": pl.Int32, "timestamp": pl.Int64}
        return pl.read_csv(file_path, schema=schema, separator=delimiter).rename({
            "user_id":"user","movie_id":"item","timestamp":"timestamp"
        })

    def _read_movies_data(self, delimiter: str) -> pl.DataFrame:
        file_path = os.path.join(self._data_dir, self._ITEMS_RAW_FILE)
        schema = {"movie_id": pl.Int32, "title": pl.Utf8, "genres": pl.Utf8}
        return pl.read_csv(file_path, schema=schema, separator=delimiter).rename({"movie_id":"item"})

    def _read_users_data(self, delimiter: str) -> pl.DataFrame:
        file_path = os.path.join(self._data_dir, self._USERS_RAW_FILE)
        schema = {"user_id": pl.Int32, "gender": pl.Utf8, "age": pl.Int32, "occupation": pl.Int32, "zip_code": pl.Utf8}
        return pl.read_csv(file_path, schema=schema, separator=delimiter).rename({"user_id":"user"})

    def _detect_encoding(self, file_name: str) -> str:
        with open(os.path.join(self._data_dir, file_name), 'rb') as file:
            raw_data = file.read(10000)  # Read the first 10 KB to guess the encoding
            result = chardet.detect(raw_data)
            return result['encoding']

    def _set_valid_delimiter(self, filename: str) -> str:
        file_path = os.path.join(self._data_dir, filename)
        new_delimiter = self._find_missing_characters(file_path)
        current_delimiter = "::"
        if new_delimiter == "":
            raise ValueError(f"Couldn't find and alternative delimiter to {current_delimiter} for {file_path}")
        new_delimiter = new_delimiter.replace('/', '\/')

        # Prepare the sed command with the -i option for in-place editing
        # For macOS, you might need to use sed -i '' for POSIX compliance
        sed_command = f"sed -i 's/{current_delimiter}/{new_delimiter}/g' {file_path}"

        # Execute the sed command
        try:
            subprocess.run(sed_command, shell=True, check=True)
            print(f"Delimiter in '{file_path}' has been replaced successfully from {current_delimiter} to {new_delimiter}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while trying to replace delimiters: {e}")
        return new_delimiter

    def _find_missing_characters(self, file_path: str) -> str:
        # Define all printable ASCII characters (extend this range for full UTF-8 as needed)
        ascii_chars = ''.join(chr(i) for i in range(32, 127))  # Printable ASCII

        # Shell command that extracts unique characters from the file
        awk_script = """{ for (i = 1; i <= length; i++) arr[substr($0, i, 1)] } END { for (a in arr) if (a == " " || a == "~" || a ~ /^[ -~]$/) print a }"""
        command = ['awk', awk_script, file_path]
        # Execute the command and capture the output
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        file_chars = set(result.stdout)
        print(f"Unique characters found in the file: {file_chars}")

        # Set of all ASCII characters
        ascii_set = set(ascii_chars)

        # Find missing characters by set difference
        missing_chars = ascii_set - file_chars
        return ''.join(sorted(missing_chars)[:1])

    def _remove_raw_data(self) -> None:
        shutil.rmtree(os.path.join(self._data_dir, "ml-1m"), ignore_errors=True)

    def __init__(
            self,
            dataset_to_use: URLZippedDataset.DatasetDescription,
            cold_start_count: int = 5,
            base_dir: Optional[str] = None
    ):
        URLZippedDataset.__init__(self, dataset_to_use, base_dir=base_dir)
        SequentialRecommenderDataset.__init__(self, cold_start_count, dataset_to_use.name, base_dir=base_dir)
