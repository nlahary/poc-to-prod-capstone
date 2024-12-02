FROM apache/airflow:2.10.0-python3.10

COPY requirements.txt /requirements.txt
COPY .dev.env /opt/airflow/.dev.env
COPY predict /opt/airflow/predict
COPY preprocessing /opt/airflow/preprocessing
COPY train /opt/airflow/train
COPY config /opt/airflow/config

ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"

USER root

RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/config && \
    chown -R airflow: /opt/airflow

USER airflow

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt
