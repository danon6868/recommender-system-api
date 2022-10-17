from fastapi import FastAPI
from web_api_utils import (
    get_recommended_feed,
    WEB_API_CONFIG,
    load_model,
    load_features,
)
from datetime import datetime
from schema import Response


app = FastAPI()


model_control = load_model("control")
features_control = load_features("control")

if WEB_API_CONFIG["models"]["test"] is None:
    model_test = None
    features_test = None
else:
    model_test = load_model("test")
    features_test = load_features("test")


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
    id: int, time: datetime, limit: int = WEB_API_CONFIG["return_posts_number"]
) -> Response:
    return get_recommended_feed(
        id,
        time,
        limit,
        model_control=model_control,
        features_control=features_control,
        model_test=model_test,
        features_test=features_test,
    )
