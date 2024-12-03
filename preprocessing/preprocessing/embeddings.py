from functools import lru_cache
import numpy as np

from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    model = TFBertModel.from_pretrained(
        settings.BERT_MODEL_PATH,
        output_hidden_states=True,
    )
    tokenizer = BertTokenizer.from_pretrained(settings.BERT_MODEL_PATH)

    return model, tokenizer


def embed(texts):
    model, tokenizer = get_embedding_model()

    embeddings = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tf.constant(tokens)[None, :]
        outputs = model(tokens)
        embeddings.append(outputs[1][0])
    return np.array(embeddings)
