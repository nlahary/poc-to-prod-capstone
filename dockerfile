FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH="/app:$PYTHONPATH"

ENV API_PORT=3000
ENV API_HOST=0.0.0.0

CMD uvicorn api.app:app --host $API_HOST --port $API_PORT   