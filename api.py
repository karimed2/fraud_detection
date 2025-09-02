from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import pandas as pd
import joblib
import numpy as np
from pydantic import validator

MODEL_PATH = "model_fraud.pkl"
NB_FEATURES = 30
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Fraud Detection API", version="1.0")

class Transaction(BaseModel):
    features: list[float]

    @validator('features')
    def check_features_length(cls, v):
        if len(v) != NB_FEATURES:
            raise ValueError(f'features doit contenir exactement {NB_FEATURES} éléments')
        return v # Sans min_items et max_items

class TransactionBatch(BaseModel):
    transactions: list[Transaction]

@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "features_expected": NB_FEATURES}

@app.post("/predict_one")
def predict_one(tx: Transaction):
    try:
        arr = np.array(tx.features).reshape(1, -1)
        pred = model.predict(arr)[0]
        result = {"prediction": int(pred)}
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(arr)[0, 1]
            result["score"] = float(score)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(batch: TransactionBatch):
    try:
        arr = np.array([t.features for t in batch.transactions])
        df = pd.DataFrame(arr)
        preds = model.predict(df)
        results = []
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(df)[:, 1]
            for p, s in zip(preds, scores):
                results.append({"prediction": int(p), "score": float(s)})
        else:
            results = [{"prediction": int(p)} for p in preds]
        return {"n": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
