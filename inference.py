import os
import torch
from transformers import DistilBertTokenizer
from model import EvoTransformerMultiTaskV3
from genome import EvoGenomeV3


# ===============================
# Device
# ===============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Label Mappings
# ===============================

TRANSACTION_LABELS = [
    "education",
    "entertainment",
    "food_grocery",
    "healthcare",
    "salary_income",
    "transfer",
    "transport",
    "utilities"
]

DOCUMENT_LABELS = [
    "business_operations",
    "financial_statement",
    "legal_regulatory",
    "management_governance",
    "risk_disclosure"
]

# Example NER labels - adjust if needed
NER_LABELS = [
    "O",
    "B-ORG",
    "I-ORG",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-MONEY",
    "I-MONEY"
]


# ===============================
# Load Tokenizer
# ===============================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# ===============================
# Load Model Safely
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "evotransformer_v31_weights.pt")

genome = EvoGenomeV3()
model = EvoTransformerMultiTaskV3(genome, 8, 5, 9)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ===============================
# Prediction Function
# ===============================

def predict(text, task):

    enc = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits, active_layers, _ = model(
            input_ids,
            attention_mask,
            task=task
        )

    # ---------------------------
    # Transaction Classification
    # ---------------------------
    if task == "transaction":
        pred_id = logits.argmax(-1).item()
        return {
            "task": "transaction",
            "prediction_id": pred_id,
            "prediction_label": TRANSACTION_LABELS[pred_id],
            "active_layers": active_layers
        }

    # ---------------------------
    # Document Classification
    # ---------------------------
    elif task == "document":
        pred_id = logits.argmax(-1).item()
        return {
            "task": "document",
            "prediction_id": pred_id,
            "prediction_label": DOCUMENT_LABELS[pred_id],
            "active_layers": active_layers
        }

    # ---------------------------
    # Named Entity Recognition
    # ---------------------------
    elif task == "ner":

        pred_ids = logits.argmax(-1)[0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        entities = []

        for token, label_id in zip(tokens, pred_ids):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                entities.append({
                    "token": token,
                    "label": NER_LABELS[label_id]
                })

        return {
            "task": "ner",
            "entities": entities,
            "active_layers": active_layers
        }

    else:
        return {
            "error": "Invalid task. Use transaction, document, or ner."
        }
