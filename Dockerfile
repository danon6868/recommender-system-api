FROM python:3.9

WORKDIR /rec_sys_web_api

COPY ./requirements.txt /rec_sys_web_api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /rec_sys_web_api/requirements.txt

COPY ./data /rec_sys_web_api/data
COPY ./training_pipeline /rec_sys_web_api/training_pipeline
COPY ./web_api /rec_sys_web_api/web_api

CMD ["uvicorn", "web_api.app:app", "--host", "0.0.0.0", "--port", "80"]
