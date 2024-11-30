FROM apache/airflow:2.10.0-python3.10

COPY requirements.txt /requirements.txt
COPY .dev.env /opt/airflow/.dev.env
COPY predict /opt/airflow/predict
COPY preprocessing /opt/airflow/preprocessing
COPY train /opt/airflow/train

ENV PYTHONPATH "${PYTHONPATH}:/opt/airflow"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt
