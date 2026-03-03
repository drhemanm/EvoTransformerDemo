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
# Label Mappings (MATCH TRAINING ORDER)
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
# Risk Mapping (Product Layer)
# ===============================

RISK_MAPPING = {
    "salary_income": "low",
    "food_grocery": "low",
    "utilities": "low",
    "education": "low",
    "healthcare": "low",
    "entertainment": "medium",
    "transport": "medium",
    "transfer": "medium"
}


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
        logits, _, _ = model(
            input_ids,
            attention_mask,
            task=task
        )

    # ==========================
    # Transaction Classification
    # ==========================
    if task == "transaction":

        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(-1).item()
        confidence = probs.max().item()

        label = TRANSACTION_LABELS[pred_id]
        risk_level = RISK_MAPPING.get(label, "unknown")

        return {
            "task": "transaction",
            "prediction_label": label,
            "confidence": round(confidence, 4),
            "risk_level": risk_level
        }

    # ==========================
    # Document Classification
    # ==========================
    elif task == "document":

        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(-1).item()
        confidence = probs.max().item()

        label = DOCUMENT_LABELS[pred_id]

        return {
            "task": "document",
            "prediction_label": label,
            "confidence": round(confidence, 4)
        }

    # ==========================
    # Named Entity Recognition
    # ==========================
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
            "entities": entities
        }

    # ==========================
    # Invalid Task
    # ==========================
    else:
        return {
            "error": "Invalid task. Use transaction, document, or ner."
        }
