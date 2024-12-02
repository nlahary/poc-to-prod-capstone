import logging
from typing import Optional
from preprocessing.preprocessing.embeddings import embed
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException
from collections import OrderedDict
from functools import lru_cache
from pydantic import BaseModel
import numpy as np
from pathlib import Path
from config import settings

# load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / '.dev.env')

# VERSION = os.getenv("API_VERSION")
# ARTEFACTS_PATH = os.getenv("ARTEFACTS_PATH")
# API_HOST = "0.0.0.0" if os.getenv("HOSTNAME") else os.getenv("API_HOST")
# API_PORT = int(os.getenv("API_PORT", 3000))

# logs_dir = Path('/app/logs')
# log_file = logs_dir / 'api.log'
# logs_dir.mkdir(parents=True, exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),  # To display logs in the console
#         # To save logs in a file
#         logging.FileHandler(
#             filename=str(log_file)  # Use the full file path
#         )
#     ]
# )

app = FastAPI()


@lru_cache()
def get_model():
    """ Load the model once and cache it """
    try:
        from run import TextPredictionModel
        return TextPredictionModel.from_artefacts(
            Path(settings.ARTEFACTS_PATH) / "2024-12-01-08-54-59"
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f'Model not found: {e}')
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f'Error loading model: {e}')


class PredictionRequest(BaseModel):
    title: list[str]


class PredictionResponse(BaseModel):
    title: list[str]
    predicted_tag: list[list[str]]


class ModelSummary(BaseModel):
    layers: list[tuple[str, tuple[Optional[int], int]]]
    trainable_params: int
    non_trainable_params: int


@ app.get("/")
async def root():
    logging.info("Redirecting to /docs")
    return RedirectResponse(url="/docs")


@ app.get('/model')
async def get_model_info():
    logging.info("Getting model summary")
    model = get_model()
    return ModelSummary(
        layers=[(layer.name, layer.output_shape)
                for layer in model.model.layers],
        trainable_params=model.model.count_params(),
        non_trainable_params=sum(
            [np.prod(layer.output_shape[1:]) for layer in model.model.layers])
    )


@ app.post(f"/{settings.API_VERSION}/predict", response_model=PredictionResponse)
async def predict_tag(request: PredictionRequest):
    logging.info(f"Predicting tags for: {request.title}")
    model = get_model()
    logging.info("Model loaded successfully")
    try:
        if isinstance(request.title, str):
            request.title = [request.title]
        logging.info(f"Embedding: {request.title}")
        embeddings = embed(request.title)
        logging.info(f"Predicting: {request.title}")
        top_5_predictions: list[OrderedDict] = model.model.predict(embeddings)
        top_k_tags = []
        for prediction in top_5_predictions:
            top_k_tags.append([model.labels_index_inv[ind]
                               for ind in np.argsort(prediction)[::-1][:5]])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f'Error predicting tags: {e}')

    return PredictionResponse(
        title=request.title,
        predicted_tag=top_k_tags
    )

if __name__ == "__main__":

    import uvicorn
    uvicorn.run("app:app",
                host=settings.API_HOST,
                port=settings.API_PORT,
                reload=True)
