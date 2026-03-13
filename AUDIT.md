# Code Repository Audit — EvoTransformerDemo

**Date:** 2026-03-13
**Scope:** Full codebase audit covering security, code quality, architecture, deployment, and maintainability.

---

## 1. Executive Summary

EvoTransformerDemo is a FastAPI-based compliance intelligence API serving a custom multi-task transformer model (transaction classification, document classification, NER) with online learning from user feedback. The codebase is compact (~650 lines of Python across 5 files) and generally well-structured, but has several issues ranging from **security vulnerabilities** to **missing tests** and **deployment concerns** that should be addressed before production use.

**Severity breakdown:**
- Critical: 2
- High: 5
- Medium: 6
- Low: 4

---

## 2. Security Issues

### CRITICAL: Unsafe `torch.load` usage (`inference.py:88-90`)

```python
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
```

`torch.load` uses Python `pickle` under the hood, which can execute arbitrary code during deserialization. If an attacker can replace the weights file, they achieve remote code execution.

**Fix:** Use `torch.load(..., weights_only=True)` (available since PyTorch 1.13+) or use `safetensors` format instead.

### CRITICAL: Online learning endpoint is a denial-of-service vector (`feedback.py:113-170`, `main.py:45-65`)

The `/feedback` endpoint triggers synchronous model training (`_micro_train`) within the request handler. An attacker with a valid API key can:
- Flood feedback to continuously retrain the model, consuming CPU/memory
- Poison the model with adversarial labels (no rate limiting, no validation of label correctness)
- Block other requests during training (the lock in `_micro_train` serializes all training)

**Fix:** Add rate limiting, move training to a background worker, and consider human review of feedback before applying.

### HIGH: Health check endpoint requires authentication (`main.py:72-75`, `render.yaml:6`)

```yaml
healthCheckPath: /learning-stats
```

The `/learning-stats` endpoint requires an API key, but Render's health check probe won't send it. This means health checks will always fail (401), potentially causing the service to be marked unhealthy and restarted in a loop.

**Fix:** Either remove auth from the health check endpoint or create a dedicated unauthenticated `/health` endpoint.

### HIGH: No input validation or size limits (`main.py:13-15`)

The `AnalyseRequest` and `FeedbackRequest` models accept arbitrary-length strings. A very long `text` field could cause excessive memory usage during tokenization/inference.

**Fix:** Add `max_length` constraints to Pydantic models:
```python
class AnalyseRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    task: str = Field(..., pattern="^(transaction|document|ner)$")
```

### HIGH: Task validation inconsistency (`main.py:36-38` vs `main.py:49-53`)

The `/analyse` endpoint does **no task validation** — it passes the raw task string directly to the model, which will raise an unhandled `ValueError` for invalid tasks. The `/feedback` endpoint properly validates. This inconsistency means `/analyse` will return a 500 error instead of a 400 for bad task values.

**Fix:** Add task validation to `/analyse` or create a shared validator.

### HIGH: No CORS configuration (`main.py`)

The `package.json` suggests a frontend exists, but no CORS middleware is configured. Browser-based clients will be blocked.

**Fix:** Add FastAPI CORS middleware with appropriate origin restrictions.

### HIGH: Model weights file committed to git

The `evotransformer_v31_weights.pt` file (16.6 MB) is tracked in git. This bloats the repository, makes cloning slow, and makes it harder to audit weight file changes.

**Fix:** Use Git LFS, or store weights in cloud storage (S3, GCS) and download at build/startup time.

---

## 3. Code Quality Issues

### MEDIUM: Duplicated label definitions (`inference.py:20-49`, `feedback.py:49-62`)

Label lists (`TRANSACTION_LABELS`, `DOCUMENT_LABELS`, `NER_LABELS`) are defined in both `inference.py` and `feedback.py`. If one is updated without the other, the model will produce incorrect predictions or fail training.

**Fix:** Define labels once (e.g., in `genome.py` or a shared `constants.py`) and import everywhere.

### MEDIUM: Magic numbers throughout

- `model = EvoTransformerMultiTaskV3(genome, 8, 5, 9)` — (`inference.py:83`) the 8, 5, 9 should be derived from the label lists, not hardcoded.
- `vocab_size=30522` in `model.py:71` — hardcoded DistilBERT vocab size.
- `max_seq_len=128` in `model.py:71` — should match tokenizer's `max_length` in `inference.py:109`.

**Fix:** Derive these from a single source of truth.

### MEDIUM: NER feedback applies single label to all tokens (`feedback.py:144-147`)

```python
target = torch.full((1, seq_len), label_idx, dtype=torch.long, device=self.device)
```

For NER tasks, the feedback system applies the `correct_label` to **every token** in the sequence. This is semantically wrong — NER labels are per-token. This will degrade NER performance over time.

**Fix:** The feedback API for NER should accept per-token labels or at minimum token-span annotations.

### MEDIUM: Thread safety gap in `_micro_train` (`feedback.py:113-170`)

The `_micro_train` method holds `self.lock` during training, but `add_feedback` (which calls `_micro_train`) does not acquire the same lock before checking buffer size and triggering training. This creates a race condition where multiple concurrent requests could each trigger `_micro_train`.

**Fix:** Acquire the lock before the buffer size check in `add_feedback`, or use a separate flag to prevent concurrent training.

### MEDIUM: No error handling around model inference (`inference.py:106-191`)

If the model raises an exception (e.g., CUDA OOM, tensor shape mismatch), the error propagates as an unhandled 500 error with potentially sensitive stack trace information.

**Fix:** Wrap inference in try/except and return a clean error response.

### MEDIUM: Unused imports and configuration (`genome.py:14`, `feedback.py:7`)

- `genome.py`: `connectivity` field and `confidence_threshold` are defined but never used anywhere in the codebase.
- `feedback.py` imports `DistilBertTokenizer` at the top level but the `OnlineLearner` receives it as a constructor parameter — the import is used only for type context.

---

## 4. Architecture Concerns

### LOW: Synchronous training blocks the event loop

FastAPI runs on an async event loop (uvicorn), but all endpoints are defined as synchronous (`def` not `async def`). While FastAPI handles sync endpoints in a threadpool, the `_micro_train` lock means training blocks one threadpool worker. Under load, this can exhaust the threadpool.

**Fix:** Move training to a background task (`BackgroundTasks`) or a separate worker process.

### LOW: No model versioning

There's no way to track which version of the model is serving predictions. After online learning, the model silently diverges from the base weights with no rollback mechanism.

**Fix:** Add model version tracking, checkpointing with timestamps, and the ability to rollback to base weights.

### LOW: Single-worker deployment

The Dockerfile runs uvicorn with default settings (1 worker). The in-memory model and feedback buffer aren't designed for multi-worker deployment.

**Fix:** This is acceptable for demo/free-tier use, but document this limitation.

### LOW: Frontend configuration orphaned

`package.json` defines a Next.js frontend but there are no frontend source files (no `pages/`, `app/`, or `src/` directories). The file appears to be a leftover or placeholder.

**Fix:** Either add the frontend or remove `package.json`.

---

## 5. Deployment & Operations

| Aspect | Status | Notes |
|--------|--------|-------|
| Docker build | OK | Clean, minimal image |
| Health check | BROKEN | Auth-gated endpoint will fail Render health probes |
| Environment secrets | OK | API key via env var, not hardcoded |
| .dockerignore | OK | Excludes `.env`, `__pycache__`, `.git` |
| Dependency pinning | PARTIAL | Only `torch` is pinned; `fastapi`, `uvicorn`, `transformers`, `pydantic` are unpinned |
| Logging | MISSING | No structured logging anywhere |
| Tests | MISSING | Zero test files |
| CI/CD | MISSING | No GitHub Actions, no linting, no automated checks |

---

## 6. Dependency Audit

| Package | Pinned? | Risk |
|---------|---------|------|
| `fastapi` | No | Minor versions could introduce breaking changes |
| `uvicorn` | No | Same |
| `transformers` | No | Major library; unpinned versions may change tokenizer behavior |
| `pydantic` | No | v1 vs v2 has major breaking changes |
| `torch==2.10.0+cpu` | Yes | Properly pinned with CPU-only variant |

**Fix:** Pin all dependencies to specific versions (e.g., `fastapi==0.115.0`).

---

## 7. Recommendations (Priority Order)

1. **Fix `torch.load` to use `weights_only=True`** — trivial fix, eliminates RCE risk
2. **Add an unauthenticated `/health` endpoint** — unblocks health checks on Render
3. **Add input validation** (text length limits, task enum) to `/analyse`
4. **Pin all dependencies** in `requirements.txt`
5. **Deduplicate label definitions** into a single module
6. **Add rate limiting** to the `/feedback` endpoint
7. **Fix NER feedback** to accept per-token labels
8. **Add basic tests** — at minimum: model loading, prediction for each task, label validation
9. **Add CORS middleware** if the frontend will call the API
10. **Add structured logging** for observability
11. **Move model weights out of git** to LFS or cloud storage
12. **Remove orphaned `package.json`** or add the frontend it references

---

## 8. What's Done Well

- Clean separation of concerns (model / inference / feedback / API / config)
- Thread-safe feedback buffer with proper locking patterns
- Constant-time API key comparison (`secrets.compare_digest`) prevents timing attacks
- Gradient clipping in online learning prevents catastrophic weight updates
- Early-exit mechanism in the model for inference efficiency
- Docker + Render deployment configuration is straightforward
- The genome/config dataclass pattern is clean and extensible
