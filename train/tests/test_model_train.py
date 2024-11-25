import unittest
from unittest.mock import MagicMock, patch
import tempfile

import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils

import os


def load_dataset_mock(filename, min_samples_per_label):
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):

    @patch("preprocessing.preprocessing.utils.LocalTextCategorizationDataset.load_dataset", side_effect=load_dataset_mock)
    def test_train(self, load_dataset_mock):
        params = {
            "batch_size": 2,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 1,
            "verbose": 1
        }

        with tempfile.TemporaryDirectory() as model_dir:
            accuracy, _ = run.train(
                "dummy.csv",
                params,
                model_dir,
                add_timestamp=False
            )
            self.assertEqual(accuracy, 1.0)
            self.assertTrue(os.path.exists(
                os.path.join(model_dir, "model.h5")))
            self.assertTrue(os.path.exists(
                os.path.join(model_dir, "params.json")))
            self.assertTrue(os.path.exists(
                os.path.join(model_dir, "train_output.json")))
            self.assertTrue(os.path.exists(
                os.path.join(model_dir, "labels_index.json")))
