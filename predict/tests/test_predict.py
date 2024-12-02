import unittest

from predict.predict import run
from config import settings
from pathlib import Path


class PredictTest(unittest.TestCase):
    def test_from_artefacts(self):
        model_timestamp = "2024-12-01-08-54-59"
        model = run.TextPredictionModel.from_artefacts(
            Path(settings.ARTEFACTS_PATH) / model_timestamp)

        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.params)
        self.assertIsNotNone(model.labels_to_index)

    def test_predict_equal_length(self):
        model_timestamp = "2024-12-01-08-54-59"
        model = run.TextPredictionModel.from_artefacts(
            Path(settings.ARTEFACTS_PATH) / model_timestamp)

        text_list = [
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            "Is it possible to execute the procedure of a function in the scope of the caller?",
            "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        ]
        top_k = 5
        tags = model.predict(text_list, top_k)
        self.assertEqual(len(tags), len(text_list))
        for tag in tags:
            self.assertEqual(len(tag), top_k)
