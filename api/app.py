from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from preprocessing.preprocessing.embeddings import embed
from api.utils import load_model, load_labels
from dotenv import load_dotenv
import os

load_dotenv()

VERSION = os.getenv("API_VERSION")

app = FastAPI()
model = load_model()
index_to_label = load_labels()


class PredictionRequest(BaseModel):
    title: str


class PredictionResponse(BaseModel):
    title: str
    predicted_tag: str
    top5_and_scores: list[tuple[str, float]]


@app.get("/")
async def root():
    return {"Model": "Text Categorization", "Version": VERSION}


@app.post(f"/{VERSION}/predict", response_model=PredictionResponse)
async def predict_tag(request: PredictionRequest):
    if not request.title:
        raise HTTPException(
            status_code=400, detail="Le titre est obligatoire")

    try:
        title_embedding = embed([request.title])
        title_embedding = np.array(title_embedding)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant le prétraitement : {str(e)}")

    try:
        predictions = model.predict(title_embedding)
        # print the top 5 labels predicted, use index_to_label to get the label name
        top5_indices = np.argsort(predictions[0])[::-1][:5]
        top5_and_scores = [(index_to_label[index], predictions[0][index])
                           for index in top5_indices]

        predicted_index = np.argmax(predictions[0])
        predicted_label = index_to_label[predicted_index]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant la prédiction : {str(e)}")

    return PredictionResponse(title=request.title, predicted_tag=predicted_label, top5_and_scores=top5_and_scores)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host=os.getenv("API_HOST", "localhost"),
                port=int(os.getenv("API_PORT", 8000))
                )
