import os
import shutil
from enum import Enum
from typing import Dict, Tuple, Optional
import polars as pl

from caser_datasets.sequential_recommender import SequentialRecommenderDataset
from caser_datasets.url_zipped import URLZippedDataset
from caser_datasets.utils import DatasetDescription

class FoursquareDataset(URLZippedDataset, SequentialRecommenderDataset):
    class Datasets(Enum):
        #TODO add other datasets
        FOURSQUARE_GLOBAL_CHECK_IN = DatasetDescription(
            name = "FourSquareGlobalScaleCheckIn",
            url = "https://drive.google.com/file/d/0BwrgZ-IdrTotZ0U0ZER2ejI3VVk/view?resourcekey=0-rlHp_JcRyFAxN7v5OAGldw",
            mirror_url = "https://drive.google.com/uc?id=1oDfzhVRtiqowqhWrUPutRwgXI9tUTV3m"
        )

    _ITEMS_RAW_FILE: str = 'dataset_TIST2015_POIs.txt'
    _DATA_RAW_FILE: str = 'dataset_TIST2015_Checkins.txt'
    _METADATA_RAW_FILE: str = 'dataset_TIST2015_Cities.txt'

    def _preprocess_data_inner(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, pl.DataFrame]]:
        city_data, items, data = self._read_city_data(), self._read_poi_data(), self._read_checkins_data()
        all_city_types, all_venue_categories, all_country_codes, all_venues = self._create_unique_mappings(city_data, items, data)
        city_data, items, data = self._join_and_clean_data(city_data, items, data, all_city_types, all_venue_categories, all_country_codes, all_venues)

        metadata = {
            "city": city_data,
            "city_types": all_city_types,
            "venue_categories": all_venue_categories,
            "country_codes": all_country_codes,
            "venues": all_venues
        }

        users = data.select("user").unique()
        return users, items, data, metadata
    
    def _remove_raw_data(self) -> None:
        self._cleanup_directory()

    def _read_city_data(self) -> pl.DataFrame:
        file_path = os.path.join(self._data_dir, self._METADATA_RAW_FILE)
        schema = {"city_name": pl.Utf8, "latitude": pl.Float32, "longitude": pl.Float32, "country_code": pl.Utf8, "country_name": pl.Utf8, "city_type": pl.Utf8}
        return pl.read_csv(file_path, schema=schema, separator="\t")

    def _read_poi_data(self) -> pl.DataFrame:
        file_path = os.path.join(self._data_dir, self._ITEMS_RAW_FILE)
        schema = {"venue_id": pl.Utf8, "latitude": pl.Float32, "longitude": pl.Float32, "venue_category": pl.Utf8, "country_code": pl.Utf8}
        return pl.read_csv(file_path, schema=schema, separator="\t")

    def _read_checkins_data(self) -> pl.DataFrame:
        file_path = os.path.join(self._data_dir, self._DATA_RAW_FILE)
        schema = {"user_id": pl.Int32, "venue_id": pl.Utf8, "utc_time": pl.Utf8, "timezone_offset_minutes": pl.Int32}
        df = pl.read_csv(file_path, schema=schema, separator="\t").rename({"user_id":"user","utc_time":"timestamp"})
        return df.with_columns(pl.col("timestamp").str.to_datetime("%a %b %d %H:%M:%S %z %Y", strict=False).dt.timestamp().alias("timestamp"))

    def _create_unique_mappings(self, df_city: pl.DataFrame, df_poi: pl.DataFrame, df_ci: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        df = df_city.select("city_type").unique()
        all_city_types = df.with_columns(pl.arange(0, df.height, eager=True).alias("city_type_numeric"))
        df = df_poi.select("venue_category").unique()
        all_venue_categories = df.with_columns(pl.arange(0, df.height, eager=True).alias("venue_category_numeric"))
        df = pl.concat([df_poi.select("country_code"), df_city.select("country_code")]).unique()
        all_country_codes = df.with_columns(pl.arange(0, df.height, eager=True).alias("country_code_numeric"))
        df = pl.concat([df_poi.select("venue_id"), df_ci.select("venue_id")]).unique()
        all_venues = df.with_columns(pl.arange(0, df.height, eager=True).alias("item"))
        return all_city_types, all_venue_categories, all_country_codes, all_venues

    def _join_and_clean_data(self, df_city: pl.DataFrame, df_poi: pl.DataFrame, df_ci: pl.DataFrame, all_city_types: pl.DataFrame, all_venue_categories: pl.DataFrame, all_country_codes: pl.DataFrame, all_venues: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        df_ci = df_ci.join(all_venues, on="venue_id", how="left").drop("venue_id")
        df_poi = df_poi.join(all_venues, on="venue_id", how="left").drop("venue_id").join(all_venue_categories, on="venue_category", how="left").drop("venue_category").join(all_country_codes, on="country_code", how="left").drop("country_code")
        df_city = df_city.join(all_country_codes, on="country_code", how="left").drop(["country_code", "country_name"]).join(all_city_types, on="city_type", how="left").drop("city_type")
        return df_city, df_poi, df_ci

    def _cleanup_directory(self) -> None:
        for filename in os.listdir(self._data_dir):
            if filename.startswith("dataset_"):
                os.remove(os.path.join(self._data_dir, filename))
        shutil.rmtree(os.path.join(self._data_dir, "__MACOSX"), ignore_errors=True)

    def __init__(
            self,
            dataset_to_use: DatasetDescription,
            cold_start_count: int = 5,
            base_dir: Optional[str] = None, **kwargs
    ):
        URLZippedDataset.__init__(self, dataset_to_use, base_dir=base_dir)
        SequentialRecommenderDataset.__init__(self, cold_start_count, dataset_to_use, base_dir=base_dir, **kwargs)