import os
import warnings
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import yaml
from loguru import logger

from common_utils import TRAINING_CONFIG, timeit
from data_handle_utils import LoadedData, extract_text_features


class DataHandler:
    def __init__(self) -> None:
        self.config = TRAINING_CONFIG
        self.loaded_data = None
        self.whole_dataset = None

    @timeit
    def load_data(self):
        if self.config["pg_connection"] is None:
            self.loaded_data = self._load_data_locally()
        else:
            self.loaded_data = self._load_data_pgsql()

    @timeit
    def save_data(self):
        if self.config["pg_connection"] is None:
            self._save_data_locally()
        else:
            self._save_data_pgsql()

    @timeit
    def extract_text_features(self):
        text_features = extract_text_features(
            self.loaded_data.posts_info["text"],
            method=TRAINING_CONFIG["text_embeddings"],
        )
        self.loaded_data.posts_info = pd.concat(
            (self.loaded_data.posts_info, text_features), axis=1
        )

    @timeit
    def create_whole_dataset(self):
        whole_dataset = pd.merge(
            self.loaded_data.feed_data,
            self.loaded_data.posts_info,
            on="post_id",
            how="left",
        )
        whole_dataset = pd.merge(
            whole_dataset, self.loaded_data.user_info, on="user_id", how="left"
        )
        whole_dataset["hour"] = pd.to_datetime(whole_dataset["timestamp"]).apply(
            lambda x: x.hour
        )
        whole_dataset["month"] = pd.to_datetime(whole_dataset["timestamp"]).apply(
            lambda x: x.month
        )
        whole_dataset = whole_dataset.drop(
            [
                "action",
                "text",
            ],
            axis=1,
        )
        whole_dataset = whole_dataset.set_index(["user_id", "post_id"])
        self.whole_dataset = whole_dataset

    @timeit
    def run(self) -> None:
        logger.info("Load data.")
        self.load_data()

        logger.info("Extract text features.")
        self.extract_text_features()

        logger.info("Create training dataset.")
        self.create_whole_dataset()

        logger.info("Finished data loading and feature extraction.")

    def _load_data_pgsql(self) -> LoadedData:
        # There is LIMIT = 100000 for fast testing
        # it was removed during the work
        feed_data = pd.read_sql(
            """SELECT * FROM public.feed_data LIMIT 100000""",
            con=self.config["pg_connection"],
        )
        user_info = pd.read_sql(
            """SELECT * FROM public.user_data""", con=self.config["pg_connection"]
        )
        posts_info = pd.read_sql(
            """SELECT * FROM public.post_text_df""", con=self.config["pg_connection"]
        )
        loaded_data = LoadedData(
            **{"feed_data": feed_data, "user_info": user_info, "posts_info": posts_info}
        )

        return loaded_data

    def _load_data_locally(self) -> LoadedData:
        self._check_local_data()
        feed_data = pd.read_csv(
            os.path.join(self.config["local_data_storage"], "feed_data.csv"),
            index_col=0,
        )
        user_info = pd.read_csv(
            os.path.join(self.config["local_data_storage"], "user_info.csv"),
            index_col=0,
        )
        posts_info = pd.read_csv(
            os.path.join(self.config["local_data_storage"], "posts_info.csv"),
            index_col=0,
        )
        loaded_data = LoadedData(
            **{"feed_data": feed_data, "user_info": user_info, "posts_info": posts_info}
        )

        return loaded_data

    def _save_data_locally(self):
        logger.info("Save data locally.")
        available_local_data = os.listdir(TRAINING_CONFIG["local_data_storage"])
        if "feed_data.csv" not in available_local_data:
            self.loaded_data.feed_data.to_csv(
                os.path.join(TRAINING_CONFIG["local_data_storage"], "feed_data.csv")
            )
        else:
            warnings.warn(
                "feed_data is already in data folder. New table will not be saved.",
                UserWarning,
            )

        if "user_info.csv" not in available_local_data:
            self.loaded_data.feed_data.to_csv(
                os.path.join(TRAINING_CONFIG["local_data_storage"], "user_info.csv")
            )
        else:
            warnings.warn(
                "user_info is already in data folder. New table will not be saved.",
                UserWarning,
            )

        if "posts_info.csv" not in available_local_data:
            logger.info(self.load_data.posts_info)
            self.loaded_data.posts_info.to_csv(
                os.path.join(TRAINING_CONFIG["local_data_storage"], "posts_info.csv")
            )
        else:
            warnings.warn(
                "posts_info is already in data folder. New table will not be saved.",
                UserWarning,
            )

        if (
            f"posts_info_{TRAINING_CONFIG['text_embeddings']}.csv"
            not in available_local_data
        ):
            self.loaded_data.posts_info.to_csv(
                os.path.join(
                    TRAINING_CONFIG["local_data_storage"],
                    f"posts_info_{TRAINING_CONFIG['text_embeddings']}.csv",
                )
            )
        else:
            warnings.warn(
                f"posts_info_{TRAINING_CONFIG['text_embeddings']} is already in data folder. New table will not be saved.",
                UserWarning,
            )

    def _save_data_pgsql(self):
        logger.info("Save data on PgSQL server.")
        logger.info(f"posts_info_features_{TRAINING_CONFIG['text_embeddings']}")
        self.loaded_data.posts_info.to_sql(
            f"posts_info_features_{TRAINING_CONFIG['text_embeddings']}",
            con=self.config["pg_connection"],
            schema="public",
            if_exists="replace",
        )

    def _read_config(self) -> Dict[str, Union[str, float, int]]:
        with open(self.config_path) as file:
            config = yaml.load(file, yaml.FullLoader)

        return config

    def _check_local_data(self):
        if "local_data_storage" not in self.config:
            raise ValueError(
                "No local data storage is provided as well as no PgSQL connection given."
            )

        if self.config["local_data_storage"] is None:
            raise ValueError(
                "Your local storage path is None as well as no PgSQL connection given."
            )

        if "feed_data.csv" not in os.listdir(self.config["local_data_storage"]):
            raise FileNotFoundError(
                "No feed_data was found in your local data storage."
            )

        if "user_info.csv" not in os.listdir(self.config["local_data_storage"]):
            raise FileNotFoundError(
                "No user_info was found in your local data storage."
            )

        if "posts_info.csv" not in os.listdir(self.config["local_data_storage"]):
            raise FileNotFoundError(
                "No posts_info was found in your local data storage."
            )
