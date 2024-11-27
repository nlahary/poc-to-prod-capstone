from preprocessing.preprocessing.embeddings import embed
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException
from collections import OrderedDict
from utils import ARTEFACT_PATH
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

import numpy as np
import os

load_dotenv()
VERSION = os.getenv("API_VERSION")

app = FastAPI()


@lru_cache()
def get_model():
    import run
    return run.TextPredictionModel.from_artefacts(
        ARTEFACT_PATH / "2024-11-27-08-32-43"
    )


class PredictionRequest(BaseModel):
    title: List[str]


class PredictionResponse(BaseModel):
    title: List[str]
    predicted_tag: List[List[str]]


@ app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@ app.post(f"/{VERSION}/predict", response_model=PredictionResponse)
async def predict_tag(request: PredictionRequest):
    print('request:', request)
    model = get_model()
    if isinstance(request.title, str):
        request.title = [request.title]
    embeddings = embed(request.title)
    top_5_predictions: list[OrderedDict] = model.model.predict(embeddings)
    top_k_tags = []
    for prediction in top_5_predictions:
        top_k_tags.append([model.labels_index_inv[ind]
                          for ind in np.argsort(prediction)[::-1][:5]])
    return PredictionResponse(
        title=request.title,
        predicted_tag=top_k_tags
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app",
                host=os.getenv("API_HOST", "localhost"),
                port=int(os.getenv("API_PORT", 3000)),
                reload=True)
