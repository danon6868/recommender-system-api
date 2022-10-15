import os
from loguru import logger
import pandas as pd
import warnings
from sklearn import metrics
from sklearn.exceptions import NotFittedError
from catboost import CatBoostClassifier
from common_utils import TRAINING_CONFIG, timeit
from model_training_utils import select_hyperparameters


class ModelTrainer:
    def __init__(self, whole_dataset: pd.DataFrame) -> None:
        self.config = TRAINING_CONFIG
        self.model = CatBoostClassifier
        self.fitted_model = None
        self.whole_dataset = whole_dataset
        self.train_size = TRAINING_CONFIG["training_params"]["train_size"]
        self.select_hyperparameters = TRAINING_CONFIG["training_params"]["select_hp"]
        self.catboost_params = None
        self.metric_function = metrics.__getattribute__(
            TRAINING_CONFIG["training_params"]["metric"]
        )

    def train(self) -> "ModelTrainer":
        if self.fitted_model is not None:
            warnings.warn(
                "You already have fitted model in `self.fitted_model`. It will be overwritten.",
                UserWarning,
            )

        X_train, X_test, y_train, y_test = self._train_test_split()
        logger.info(
            f"The number of train samples: {y_train.shape[0]}, and test samples: {y_test.shape[0]}"
        )
        default_catboost_params = TRAINING_CONFIG["training_params"][
            "default_catboost_params"
        ]

        if self.select_hyperparameters:
            best_params = select_hyperparameters(
                self.model, X_train, X_test, y_train, y_test
            )
            best_params.update(default_catboost_params)
            self.catboost_params = best_params

        else:
            self.catboost_params = default_catboost_params

        model = self.model(
            cat_features=TRAINING_CONFIG["training_params"]["object_cols"],
            **self.catboost_params,
        )
        model.fit(X_train, y_train)
        logger.info(
            f"Quality on train: {self.metric_function(y_train, model.predict_proba(X_train)[:, 1])}"
        )
        logger.info(
            f"Quality on test: {self.metric_function(y_test, model.predict_proba(X_test)[:, 1])}"
        )

        if TRAINING_CONFIG["training_params"]["retrain_whole_dataset"]:
            logger.info("Retraining the model the on whole dataset.")
            X = pd.concat([X_train, X_test], axis=0)
            y = pd.concat([y_train, y_test], axis=0)
            model.fit(X, y)

        self.fitted_model = model

        return self

    def save(self):
        if self.fitted_model is None:
            raise NotFittedError(
                "Your should run train method before saving the model."
            )
        model_save_path = os.path.join(
            "training_results", f"catboost_{TRAINING_CONFIG['text_embeddings']}"
        )
        logger.info(f"Saving model in {model_save_path}")
        self.fitted_model.save_model(model_save_path, format="cbm")

    def _train_test_split(self):
        threshold_date_index = int(self.whole_dataset.shape[0] * self.train_size)
        sorted_timestamps = self.whole_dataset["timestamp"].sort_values().values
        threshold_date = sorted_timestamps[threshold_date_index]

        train_dataset = self.whole_dataset[
            self.whole_dataset["timestamp"] < threshold_date
        ]
        test_dataset = self.whole_dataset[
            self.whole_dataset["timestamp"] >= threshold_date
        ]

        train_dataset = train_dataset.drop("timestamp", axis=1)
        test_dataset = test_dataset.drop("timestamp", axis=1)

        X_train = train_dataset.drop("target", axis=1)
        X_test = test_dataset.drop("target", axis=1)

        y_train = train_dataset["target"]
        y_test = test_dataset["target"]

        return X_train, X_test, y_train, y_test
