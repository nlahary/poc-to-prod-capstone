from pathlib import Path
from predict.predict import run

ARTEFACT_PATH = Path(__file__).resolve(
    strict=True).parent.parent.parent / "train/data/artefacts"

model_timestamp = "2024-11-24-20-50-29"

model = run.TextPredictionModel.from_artefacts(
    ARTEFACT_PATH / model_timestamp)

print(model.model)
print(model.params)
print(model.labels_to_index)
