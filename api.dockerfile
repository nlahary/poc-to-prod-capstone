FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt predict preprocessing train config .dev.env ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt