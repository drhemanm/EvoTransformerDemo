import os
import time
import secrets
import logging
from collections import defaultdict
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from inference import predict, _model_ready
import inference

# ===============================
# Logging
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("evocompliance")

# ===============================
# App
# ===============================

app = FastAPI(title="EvoCompliance API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("EVO_API_KEY")

# ===============================
# Request Tracking (Analytics)
# ===============================

request_log: list[dict] = []
MAX_REQUEST_LOG = 1000

def log_request(task: str, confidence: float | None = None, source: str = "analyse"):
    entry = {"task": task, "confidence": confidence, "source": source, "timestamp": time.time()}
    request_log.append(entry)
    if len(request_log) > MAX_REQUEST_LOG:
        request_log.pop(0)

# ===============================
# Rate Limiting
# ===============================

rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30  # requests per window


def _check_rate_limit(client_id: str):
    now = time.time()
    hits = rate_limit_store[client_id]
    # Prune old entries
    rate_limit_store[client_id] = [t for t in hits if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[client_id]) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    rate_limit_store[client_id].append(now)


# ===============================
# Request Models (with validation)
# ===============================

VALID_TASKS = ("transaction", "document", "ner")


class AnalyseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    task: str = Field(..., pattern="^(transaction|document|ner)$")


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    task: str = Field(..., pattern="^(transaction|document|ner)$")
    correct_label: str = Field(..., min_length=1, max_length=100)


# ===============================
# Auth
# ===============================

def _check_auth(x_api_key: Optional[str]):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ===============================
# Health Endpoint (unauthenticated)
# ===============================

@app.get("/health")
def health():
    ready = _model_ready.is_set()
    return {"status": "ok" if ready else "warming_up", "model_ready": ready, "version": "0.3.1"}


# ===============================
# Analyse Endpoint
# ===============================

@app.post("/analyse")
def analyse(req: AnalyseRequest, request: Request, x_api_key: str = Header(None)):
    _check_auth(x_api_key)
    _check_rate_limit(x_api_key or "anonymous")

    logger.info("Analyse request: task=%s, text_len=%d", req.task, len(req.text))

    result = predict(req.text, req.task)

    log_request(req.task, result.get("confidence"), "analyse")
    logger.info("Analyse result: task=%s, label=%s, confidence=%s",
                req.task, result.get("prediction_label"), result.get("confidence"))
    return result


# ===============================
# Feedback Endpoint (Live Learning)
# ===============================

@app.post("/feedback")
def feedback(req: FeedbackRequest, request: Request, x_api_key: str = Header(None)):
    _check_auth(x_api_key)
    _check_rate_limit(x_api_key or "anonymous")

    logger.info("Feedback request: task=%s, label=%s, text_len=%d",
                req.task, req.correct_label, len(req.text))

    if not _model_ready.is_set() or inference.learner is None:
        raise HTTPException(status_code=503, detail="Model is still warming up. Retry shortly.")

    success, message = inference.learner.add_feedback(req.text, req.task, req.correct_label)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    log_request(req.task, source="feedback")
    logger.info("Feedback result: %s", message)

    return {
        "status": "accepted",
        "message": message,
        "stats": inference.learner.get_stats()
    }


# ===============================
# Learning Stats Endpoint
# ===============================

@app.get("/learning-stats")
def learning_stats(x_api_key: str = Header(None)):
    _check_auth(x_api_key)
    if not _model_ready.is_set() or inference.learner is None:
        raise HTTPException(status_code=503, detail="Model is still warming up.")
    return inference.learner.get_stats()


# ===============================
# Analytics Endpoint
# ===============================

@app.get("/analytics")
def analytics(x_api_key: str = Header(None)):
    _check_auth(x_api_key)

    now = time.time()
    last_hour = [r for r in request_log if now - r["timestamp"] < 3600]
    last_day = [r for r in request_log if now - r["timestamp"] < 86400]

    def summarize(entries: list[dict]) -> dict:
        if not entries:
            return {"total": 0, "by_task": {}, "by_source": {}, "avg_confidence": None}
        by_task: dict[str, int] = defaultdict(int)
        by_source: dict[str, int] = defaultdict(int)
        confidences = []
        for e in entries:
            by_task[e["task"]] += 1
            by_source[e["source"]] += 1
            if e.get("confidence") is not None:
                confidences.append(e["confidence"])
        return {
            "total": len(entries),
            "by_task": dict(by_task),
            "by_source": dict(by_source),
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        }

    return {
        "last_hour": summarize(last_hour),
        "last_24h": summarize(last_day),
        "learning": inference.learner.get_stats() if inference.learner else {},
    }


# ===============================
# Serve Frontend (Static Export)
# ===============================

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend_out")

if os.path.isdir(FRONTEND_DIR):
    @app.get("/app/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fallback to index.html for SPA routing
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Not found")

    @app.get("/app")
    async def serve_frontend_root():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not built")

    # Mount static assets if they exist
    next_static = os.path.join(FRONTEND_DIR, "_next")
    if os.path.isdir(next_static):
        app.mount("/app/_next", StaticFiles(directory=next_static), name="next-static")


logger.info("EvoCompliance API v0.3.0 started")
