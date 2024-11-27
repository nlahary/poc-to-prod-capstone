import unittest

from predict.predict import run
from pathlib import Path

ARTEFACT_PATH = Path(__file__).resolve(
    strict=True).parent.parent.parent / "train/data/artefacts"


class PredictTest(unittest.TestCase):
    def test_from_artefacts(self):
        model_timestamp = "2024-11-24-20-50-29"
        # print(f'ARTEFACT_PATH: {ARTEFACT_PATH / model_timestamp}')
        model = run.TextPredictionModel.from_artefacts(
            ARTEFACT_PATH / model_timestamp)

        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.params)
        self.assertIsNotNone(model.labels_to_index)

    def test_predict_equal_length(self):
        model_timestamp = "2024-11-24-20-50-29"
        model = run.TextPredictionModel.from_artefacts(
            ARTEFACT_PATH / model_timestamp)

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
