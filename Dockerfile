FROM python:3.9

WORKDIR /rec_sys_web_api

COPY recommender_system/web_api/requirements_container.txt /rec_sys_web_api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /rec_sys_web_api/requirements.txt

COPY recommender_system/data /rec_sys_web_api/data
COPY recommender_system/training_pipeline_config.yaml /rec_sys_web_api/training_pipeline_config.yaml
COPY recommender_system/web_api /rec_sys_web_api/web_api

WORKDIR /rec_sys_web_api/web_api

RUN ls

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
