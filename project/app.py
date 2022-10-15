import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger


app = FastAPI()


def batch_load_sql(query: str, chunksize: int = 200000):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    )

    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for i, chunk_dataframe in enumerate(pd.read_sql(query, conn, chunksize=chunksize)):
        chunks.append(chunk_dataframe)
        logger.info(f"Loading chunk: {i+1}")
    conn.close()

    return pd.concat(chunks, ignore_index=True)


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = "/workdir/user_input/model"
    else:
        MODEL_PATH = path

    return MODEL_PATH


def load_features():
    logger.info("Loading liked posts...")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action='like'
    """

    liked_posts = batch_load_sql(liked_posts_query)

    logger.info("Loading posts features")
    posts_features = pd.read_sql(
        """SELECT * FROM public.posts_info_features_dl_litvinov_roberta""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
    )

    logger.info("Loading user features...")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
    )

    return [liked_posts, posts_features, user_features]


def load_models():
    model_path = get_model_path(
        "/home/daniil/Documents/Education/Karpov_courses/ML/ml_hws/recommender_project/project/model"
    )
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)

    return loaded_model


logger.info("Loading model...")
model = load_models()
logger.info("Loading features...")
features = load_features()
logger.info("Service is up and running...")


def get_recommened_feed(id: int, time: datetime, limit: int):
    logger.info(f"user_id: {id}")
    logger.info(f"Reading features...")

    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop("user_id", axis=1)

    logger.info("Drop columns...")
    posts_features = features[1].drop(["index", "text"], axis=1)
    content = features[1][["post_id", "text", "topic"]]

    logger.info("Zipping everything...")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("Assigning everything")
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index("post_id")

    logger.info("Add time info...")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    logger.info("Predict...")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    logger.info("Delete liked posts...")
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    return [
        PostGet(
            **{
                "id": i,
                "text": content[content.post_id == i].text.values[0],
                "topic": content[content.post_id == i].topic.values[0],
            }
        )
        for i in recommended_posts
    ]


@app.get("/post/recommendations", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    return get_recommened_feed(id, time, limit)
