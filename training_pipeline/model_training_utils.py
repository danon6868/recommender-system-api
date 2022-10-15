import os
import joblib
import optuna
import pandas as pd
from loguru import logger
from typing import Dict, Union
from catboost import CatBoostClassifier
from sklearn import metrics
from common_utils import TRAINING_CONFIG


def select_hyperparameters(
    model_type: CatBoostClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Union[str, float, int]]:
    metric_function = metrics.__getattribute__(
        TRAINING_CONFIG["training_params"]["metric"]
    )
    logger.info(f"Using {metric_function.__name__} as metric.")

    def objective(trial):
        params_range = TRAINING_CONFIG["training_params"]["optuna_search"][
            "search_params"
        ]

        search_params = {
            "objective": trial.suggest_categorical(
                "objective", params_range["objective"]
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel",
                params_range["colsample_bylevel"][0],
                params_range["colsample_bylevel"][1],
            ),
            "depth": trial.suggest_int(
                "depth",
                params_range["depth"][0],
                params_range["depth"][1],
            ),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", params_range["boosting_type"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", params_range["bootstrap_type"]
            ),
        }

        if search_params["bootstrap_type"] == "Bayesian":
            search_params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature",
                params_range["bagging_temperature"][0],
                params_range["bagging_temperature"][1],
            )
        elif search_params["bootstrap_type"] == "Bernoulli":
            search_params["subsample"] = trial.suggest_float(
                "subsample", params_range["subsample"][0], params_range["subsample"][1]
            )

        model = model_type(
            cat_features=TRAINING_CONFIG["training_params"]["object_cols"],
            **search_params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=0,
            early_stopping_rounds=100,
        )

        preds = model.predict_proba(X_test)[:, 1]

        metric_value = metric_function(y_test, preds)

        return metric_value

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=TRAINING_CONFIG["training_params"]["optuna_search"]["n_trials"],
    )

    # Pickle study object so that plot some graphs
    study_save_path = os.path.join(
        "training_results",
        f"catboost_optuna_study_{TRAINING_CONFIG['text_embeddings']}.pkl",
    )
    logger.info(f"Saving optuna search results in {study_save_path}.")
    joblib.dump(study, study_save_path)

    best_params = study.best_params

    return best_params
