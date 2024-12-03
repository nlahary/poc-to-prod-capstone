FROM apache/airflow:2.10.0-python3.10

COPY requirements.txt predict preprocessing train config .dev.env ./

USER root

RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/config && \
    chown -R airflow: /opt/airflow

USER airflow

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
