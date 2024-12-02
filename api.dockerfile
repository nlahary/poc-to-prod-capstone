FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# download BERT model
RUN python -c "from transformers import BertTokenizer, TFBertModel; TFBertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased')"

COPY predict ./predict
COPY preprocessing ./preprocessing
COPY train ./train
COPY config ./config
COPY .dev.env .
ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "predict/predict/app.py"]
