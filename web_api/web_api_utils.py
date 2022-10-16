import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
from datetime import datetime
from loguru import logger
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
import pandas as pd
import yaml
from schema import Response, PostGet


def read_config(
    config_path: Union[str, Path] = Path("web_api_config.yaml")
) -> Dict[str, Union[str, float, int]]:
    with open(config_path) as file:
        config = yaml.load(file, yaml.FullLoader)

    return config


WEB_API_CONFIG = read_config()
TRAINING_CONFIG = read_config("../training_pipeline/training_pipeline_config.yaml")


def batch_load_sql(query: str, connection):
    engine = create_engine(connection)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for i, chunk_dataframe in enumerate(pd.read_sql(query, conn, chunksize=100000)):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chunk: {i + 1}")
        break
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_raw_features_pgsql(connection, model_version):
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


def load_raw_features_local(model_version: str) -> str:
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


def load_features(model_version):
    if TRAINING_CONFIG["pg_connection"] is None:
        features = load_raw_features_local(model_version)
    else:
        features = load_raw_features_pgsql(
            TRAINING_CONFIG["pg_connection"], model_version
        )

    return features


def get_model_path(model_version: str) -> str:
    model_path = WEB_API_CONFIG["models"][model_version]

    return model_path


def load_model(model_version: str):
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)

    return loaded_model


def get_user_group(id: int) -> str:
    value_str = str(id) + WEB_API_CONFIG["user_salt"]
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"

    elif percent < 100:
        return "test"

    return "unknown"


def calculate_features(
    features: List[pd.DataFrame], id: int, time: datetime, model_version: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Тут мы готовим фичи, при этом в зависимости от группы это могут быть
    разные фичи под разные модели. Здесь это одни и те же фичи (то есть будто
    бы разница в самих моделях)
    """
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop("user_id", axis=1)

    logger.info("dropping columns")
    posts_features = features[1].drop(["text"], axis=1)

    logger.info("zipping everything")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index("post_id")

    # Добафим информацию о дате рекомендаций
    logger.info("add time info")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    return user_features, user_posts_features


def get_recommended_feed(
    id: int,
    time: datetime,
    limit: int,
    model_control: CatBoostClassifier,
    features_control: List[pd.DataFrame],
    model_test: Union[CatBoostClassifier, None] = None,
    features_test: Union[List[pd.DataFrame], None] = None,
) -> Response:
    # Выбираем группу пользователи
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

    # Вычисляем фичи
    user_features, user_posts_features = calculate_features(
        features=features, id=id, time=time, model_version=user_group
    )

    # Сформируем предсказания вероятности лайкнуть пост для всех постов
    logger.info("Predicting")
    logger.info(model.feature_names_)
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    # Уберем записи, где пользователь ранее уже ставил лайк
    logger.info("deleting liked posts")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # Рекомендуем топ-5 по вероятности постов
    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    return Response(
        recommendations=[
            PostGet(id=i[0], text="lexa", topic=i[1])
            for i in user_features.itertuples(index=False)
        ],
        exp_group=user_group,
    )
