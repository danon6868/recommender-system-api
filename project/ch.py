from http import client
import app
from fastapi.testclient import TestClient
from datetime import datetime


client = TestClient(app.app)


user_id = 1000
time = datetime(2021, 12, 20)


try:
    r = client.get(
        f"/post/recommendations/", params={"id": user_id, "time": time, "limit": 5}
    )
except Exception as e:
    raise ValueError(f"Error during request: {type(e)} {str(e)}")


print(r.json())
