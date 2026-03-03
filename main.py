import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="EvoCompliance API", version="0.1.0")

API_KEY = os.getenv("EVO_API_KEY")

class Request(BaseModel):
    text: str
    task: str


@app.post("/analyse")
def analyse(req: Request, x_api_key: str = Header(None)):

    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return predict(req.text, req.task)
