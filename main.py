import os
import secrets
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
from inference import predict, learner

app = FastAPI(title="EvoCompliance API", version="0.2.0")

API_KEY = os.getenv("EVO_API_KEY")


class AnalyseRequest(BaseModel):
    text: str
    task: str


class FeedbackRequest(BaseModel):
    text: str
    task: str
    correct_label: str


def _check_auth(x_api_key: Optional[str]):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ===============================
# Analyse Endpoint
# ===============================

@app.post("/analyse")
def analyse(req: AnalyseRequest, x_api_key: str = Header(None)):
    _check_auth(x_api_key)
    return predict(req.text, req.task)


# ===============================
# Feedback Endpoint (Live Learning)
# ===============================

@app.post("/feedback")
def feedback(req: FeedbackRequest, x_api_key: str = Header(None)):
    _check_auth(x_api_key)

    valid_tasks = ("transaction", "document", "ner")
    if req.task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Use one of: {', '.join(valid_tasks)}"
        )

    success, message = learner.add_feedback(req.text, req.task, req.correct_label)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "accepted",
        "message": message,
        "stats": learner.get_stats()
    }


# ===============================
# Learning Stats Endpoint
# ===============================

@app.get("/learning-stats")
def learning_stats(x_api_key: str = Header(None)):
    _check_auth(x_api_key)
    return learner.get_stats()
