# EvoTransformer Demo

A lightweight, evolutionary transformer for financial compliance tasks. The model uses a genetically optimized architecture (EvoGenome) with multi-task heads for transaction classification, document classification, and named entity recognition.

**Live demo:** [evotransformerdemo.onrender.com](https://evotransformerdemo.onrender.com)

## Features

- **Multi-task inference** — a single model handles three compliance tasks:
  - **Transaction classification** — categorizes transactions (e.g. salary, transfer, utilities) with risk-level tagging
  - **Document classification** — classifies financial documents (e.g. legal/regulatory, financial statement, risk disclosure)
  - **Named Entity Recognition (NER)** — extracts entities like organizations, people, locations, and monetary amounts
- **Evolutionary architecture** — model structure (layers, heads, FFN dim, dropout, early exit) is defined by an `EvoGenomeV3` dataclass, enabling architecture search
- **Early exit** — layers have learned exit gates that skip remaining computation when confidence is high
- **Online learning** — accepts user feedback via `/feedback` endpoint and performs live micro-training without full retraining
- **Web UI** — built-in frontend served at `/` for interactive testing

## Architecture

```
DistilBERT tokenizer
        |
EvoTransformerBackboneV3  (2 layers, 128-dim, early exit gates)
        |
   +---------+
   |    |    |
  TXN  DOC  NER   <-- task-specific classification heads
```

Genome defaults: 2 layers, 128 embed dim, 512 FFN dim, CLS pooling, 0.05 dropout, early exit threshold 0.50.

## API Endpoints

All endpoints (except `/` and `/docs`) require an `x-api-key` header matching the `EVO_API_KEY` environment variable.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/docs` | Interactive API docs (Swagger) |
| `POST` | `/analyse` | Run inference on text |
| `POST` | `/feedback` | Submit correction for online learning |
| `GET` | `/learning-stats` | View online learning statistics |

### Example: Analyse

```bash
curl -X POST https://evotransformerdemo.onrender.com/analyse \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Monthly salary deposit from employer", "task": "transaction"}'
```

### Example: Feedback

```bash
curl -X POST https://evotransformerdemo.onrender.com/feedback \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Wire transfer to overseas account", "task": "transaction", "correct_label": "transfer"}'
```

## Project Structure

```
main.py          -- FastAPI app with endpoints
inference.py     -- tokenizer, model loading, prediction logic
model.py         -- EvoTransformer backbone and multi-task heads
genome.py        -- EvoGenomeV3 dataclass (architecture config)
feedback.py      -- online learning (FeedbackBuffer + OnlineLearner)
static/          -- frontend web UI
Dockerfile       -- container config for deployment
render.yaml      -- Render deployment blueprint
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export EVO_API_KEY="your-secret-key"

# Run the server
uvicorn main:app --reload --port 8000
```

Then open http://localhost:8000 for the UI or http://localhost:8000/docs for Swagger.

## Deployment

The app is configured for [Render](https://render.com) via Docker. See `render.yaml` for the blueprint. The `EVO_API_KEY` environment variable is auto-generated on first deploy.

## Label Reference

**Transaction:** education, entertainment, food_grocery, healthcare, salary_income, transfer, transport, utilities

**Document:** business_operations, financial_statement, legal_regulatory, management_governance, risk_disclosure

**NER (BIO scheme):** O, B-ORG, I-ORG, B-PER, I-PER, B-LOC, I-LOC, B-MONEY, I-MONEY
