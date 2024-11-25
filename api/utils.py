import tensorflow.keras as tf
import json
from pathlib import Path

MODEL_PATH = Path(__file__).resolve(strict=True).parent.parent / \
    "train/data/artefacts/2024-11-24-20-50-29/model.h5"
LABELS_INDEX_PATH = Path(__file__).resolve(
    strict=True).parent.parent / "train/data/artefacts/2024-11-24-20-50-29/labels_index.json"


def load_model():
    model = tf.models.load_model(MODEL_PATH)
    return model


def load_labels():
    with open(LABELS_INDEX_PATH, "r") as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    return index_to_label
