import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import yaml
from catboost import CatBoostClassifier
from loguru import logger
from sqlalchemy import create_engine

from schema import PostGet, Response


def read_config(
    config_path: Union[str, Path] = Path("web_api_config.yaml")
) -> Dict[str, Union[str, float, int]]:
    """Function for config opening.

    Args:
        config_path (Union[str, Path], optional): Path to config.yaml.

    Returns:
        Dict[str, Union[str, float, int]]: Dictionary config representation.
    """

    with open(config_path) as file:
        config = yaml.load(file, yaml.FullLoader)

    return config


WEB_API_CONFIG = read_config()
TRAINING_CONFIG = read_config("../training_pipeline/training_pipeline_config.yaml")


def batch_load_sql(query: str, connection: str) -> pd.DataFrame:
    """Function that helps to save RAM during data loading from PgSQL database.

    Args:
        query (str): SQL query.
        connection (str): Connection to PgSQL database.

    Returns:
        pd.DataFrame: Table loaded from database.
    """

    engine = create_engine(connection)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for i, chunk_dataframe in enumerate(pd.read_sql(query, conn, chunksize=100000)):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {i + 1}")
        break
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_raw_features_pgsql(connection: str, model_version: str) -> List[pd.DataFrame]:
    """Load raw tables from PgSQL database.

    Args:
        connection (str): Connection to PgSQL database.
        model_version (str): Model version can be control or test.
        This is specified in web_api_config.yaml.

    Returns:
        List[pd.DataFrame]: List of `liked_posts`, `posts_features`, `user_features` tables.
    """

    logger.info("Loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = batch_load_sql(liked_posts_query, connection)

    logger.info("Loading posts features")
    posts_features = pd.read_sql(
        f"""SELECT * FROM public.posts_info_features_{WEB_API_CONFIG['models'][f'{model_version}_posts_features']}""",
        con=connection,
    )

    logger.info("Loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con=connection,
    )

    return [liked_posts, posts_features, user_features]


def load_raw_features_local(
    model_version: str,
) -> List[pd.DataFrame]:
    """Load raw tables from local data storage.

    Args:
        model_version (str): Model version can be control or test.
        This is specified in web_api_config.yaml.

    Returns:
        List[pd.DataFrame]: List of `liked_posts`, `posts_features`, `user_features` tables.
    """

    logger.info("Loading liked posts.")
    liked_posts = pd.read_csv(
        os.path.join(TRAINING_CONFIG["local_data_storage"], "feed_data.csv"),
        index_col=0,
    )
    liked_posts = liked_posts.drop_duplicates(subset=["user_id", "post_id"]).query(
        "action == 'like'"
    )

    logger.info("Loading posts features.")
    posts_features = pd.read_csv(
        os.path.join(
            TRAINING_CONFIG["local_data_storage"],
            f"posts_info_{WEB_API_CONFIG['models'][f'{model_version}_posts_features']}.csv",
        ),
        index_col=0,
    )

    logger.info("Loading user features.")
    user_features = pd.read_csv(
        os.path.join(
            TRAINING_CONFIG["local_data_storage"],
            "user_info.csv",
        ),
        index_col=0,
    )

    return [liked_posts, posts_features, user_features]


def load_features(model_version: str) -> List[pd.DataFrame]:
    """Load raw tables for making predictions.

    Args:
        model_version (str): Model version can be control or test.
        This is specified in web_api_config.yaml.

    Returns:
        List[pd.DataFrame]: List of `liked_posts`, `posts_features`, `user_features` tables.
    """

    if TRAINING_CONFIG["pg_connection"] is None:
        features = load_raw_features_local(model_version)
    else:
        features = load_raw_features_pgsql(
            TRAINING_CONFIG["pg_connection"], model_version
        )

    return features


def get_model_path(model_version: str) -> str:
    """Construct path to model based on model version,
     and current configuration in web_api_config.yaml.

    Args:
        model_version (str): Model version can be control or test.
        This is specified in web_api_config.yaml.

    Returns:
        str: Path to model for recommendations.
    """

    model_path = WEB_API_CONFIG["models"][model_version]

    return model_path


def load_model(model_version: str) -> CatBoostClassifier:
    """Load CatBoostClassifier model.

    Args:
        model_version (str):  Model version can be control or test.
        This is specified in web_api_config.yaml.

    Returns:
        CatBoostClassifier: Loaded CatBoostClassifier model.
    """

    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)

    return loaded_model


def get_user_group(id: int) -> str:
    """Define user group (control or test) calculating hash based on user_id
    with salt (specified in web_api_config.yaml).

    Args:
        id (int): User id.

    Returns:
        str: User group.
    """

    value_str = str(id) + WEB_API_CONFIG["user_salt"]
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"

    elif percent < 100:
        return "test"

    return "unknown"


def calculate_features(
    features: List[pd.DataFrame], id: int, time: datetime
) -> Tuple[pd.DataFrame]:
    """Here we are preparing features, and depending on the group, these can be
    different features for different models.

    Args:
        features (List[pd.DataFrame]): Loaded raw tables.
        id (int): User id.
        time (datetime): Date and time when recommendation will be made.

    Returns:
        Tuple[pd.DataFrame]: The tuple of tables `user_features`, `user_posts_features`, `content`
        which will be used for making a predictions and posts recommendation.
    """

    logger.info(f"user_id: {id}")
    logger.info("Reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop("user_id", axis=1)

    # This part is used because of different table formats in local data storage and PgSQL database
    try:
        posts_features = features[1].drop(["text", "index"], axis=1)
    except KeyError:
        posts_features = features[1].drop(["text"], axis=1)

    content = features[1][["post_id", "text", "topic"]]
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index("post_id")

    # Add information about the date of recommendation
    logger.info("Add time info.")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    logger.info("Features were prepared successfully.")

    return user_features, user_posts_features, content


def get_recommended_feed(
    id: int,
    time: datetime,
    limit: int,
    model_control: CatBoostClassifier,
    features_control: List[pd.DataFrame],
    model_test: Union[CatBoostClassifier, None] = None,
    features_test: Union[List[pd.DataFrame], None] = None,
) -> Response:
    """For a given user_id and time return a limit number of recommended posts.

    Args:
        id (int): User id.
        time (datetime): Date and time when recommendation will be made.
        limit (int): The number of given recommendations.
        model_control (CatBoostClassifier): Model for control group.
        features_control (List[pd.DataFrame]): Features for control group.
        model_test (Union[CatBoostClassifier, None], optional): Model for test group.
        Defaults to None if service is used in general mode instead of A/B testing.
        features_test (Union[List[pd.DataFrame], None], optional): Features for test group.
        Defaults to None if service is used in general mode instead of A/B testing.

    Raises:
        ValueError: If got unknown user group.

    Returns:
        Response: The response object which contains information about user group
        and limit number of recommendations: post_id, topic and post text.
    """

    # Choose user group
    user_group = get_user_group(id=id)
    logger.info(f"User group {user_group}.")

    if model_test is None:
        model = model_control
        features = features_control
    else:
        if user_group == "control":
            model = model_control
            features = features_control
        elif user_group == "test":
            model = model_test
            features = features_test
        else:
            raise ValueError("Unknown group")

    # Prepare features for predictions
    user_features, user_posts_features, content = calculate_features(
        features=features, id=id, time=time
    )

    # Make predictions
    logger.info("Predicting.")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Remove posts where the user has previously liked
    logger.info("Deleting liked posts.")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Recommended top-limit by post probability
    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    return Response(
        recommendations=[
            PostGet(
                id=i,
                text=content[content.post_id == i].text.values[0],
                topic=content[content.post_id == i].topic.values[0],
            )
            for i in recommended_posts
        ],
        exp_group=user_group,
    )
