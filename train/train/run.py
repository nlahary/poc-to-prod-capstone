import os
import json
import argparse
import time
import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from preprocessing.preprocessing.embeddings import embed
from preprocessing.preprocessing.utils import LocalTextCategorizationDataset

logger = logging.getLogger(__name__)


def train(dataset_path, train_conf, model_path, add_timestamp):
    """
    :param dataset_path: path to a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
    :param train_conf: dictionary containing training parameters, example :
            {
                batch_size: 32
                epochs: 1
                dense_dim: 64
                min_samples_per_label: 10
                verbose: 1
            }
    :param model_path: path to folder where training artefacts will be persisted
    :param add_timestamp: boolean to create artefacts in a sub folder with name equal to execution timestamp
    """

    if add_timestamp:
        artefacts_path = os.path.join(
            model_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        artefacts_path = model_path

    dataset = LocalTextCategorizationDataset(
        filename=dataset_path,
        batch_size=train_conf['batch_size'],
        train_ratio=0.8,
        min_samples_per_label=train_conf['min_samples_per_label'],
        preprocess_text=embed,
        random_state=42
    )

    model = Sequential([
        Dense(
            units=train_conf['dense_dim'],
            activation='relu',
            input_shape=(768,)  # BERT embedding dimension
        ),
        Dense(
            units=dataset.get_num_labels(),
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_sequence = dataset.get_train_sequence()
    test_sequence = dataset.get_test_sequence()

    train_history = model.fit(
        train_sequence,
        epochs=train_conf['epochs'],
        verbose=train_conf['verbose'],
        validation_data=test_sequence
    )

    scores = model.evaluate(test_sequence, verbose=0)
    logger.info("Test Accuracy: {:.2f}".format(scores[1] * 100))

    os.makedirs(artefacts_path, exist_ok=True)

    model.save(os.path.join(artefacts_path, "model.h5"))

    with open(os.path.join(artefacts_path, "params.json"), "w") as f:
        json.dump(train_conf, f)

    labels_index = dataset.get_label_to_index_map()
    with open(os.path.join(artefacts_path, "labels_index.json"), "w") as f:
        json.dump(labels_index, f)

    serializable_hist = {k: [float(e) for e in v]
                         for k, v in train_history.history.items()}
    with open(os.path.join(artefacts_path, "train_output.json"), "w") as f:
        json.dump(serializable_hist, f)

    return scores[1], artefacts_path


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to training dataset")
    parser.add_argument(
        "config_path", help="Path to Yaml file specifying training parameters")
    parser.add_argument(
        "artefacts_path", help="Folder where training artefacts will be persisted")
    parser.add_argument("add_timestamp", action='store_true',
                        help="Create artefacts in a sub folder with name equal to execution timestamp")

    args = parser.parse_args()

    with open(args.config_path, 'r') as config_f:
        train_params = yaml.safe_load(config_f.read())

    logger.info(f"Training model with parameters: {train_params}")

    train(args.dataset_path, train_params,
          args.artefacts_path, args.add_timestamp)
