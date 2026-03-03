from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="EvoCompliance API")

class Request(BaseModel):
    text: str
    task: str  # "transaction" | "document" | "ner"

@app.post("/analyse")
def analyse(req: Request):
    result = predict(req.text, req.task)
    return result
