import unittest
from unittest.mock import MagicMock
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
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(
        side_effect=load_dataset_mock)

    def test_train(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            "batch_size": 2,
            "epochs": 1,
            "dense_dim": 64,
            "min_samples_per_label": 1,
            "verbose": 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, artefacts_path = run.train(
                "dummy.csv",
                params,
                model_dir,
                add_timestamp=False
            )
        print(os.listdir(artefacts_path))
        # TODO: CODE HERE
        # assert that accuracy is equal to 1.0
        self.assertEqual(accuracy, 1.0)
        # assert artefacts are created
        self.assertTrue(os.path.exists(
            os.path.join(artefacts_path, "model.h5")))
        self.assertTrue(os.path.exists(
            os.path.join(artefacts_path, "params.json")))
