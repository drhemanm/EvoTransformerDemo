import os
import random
import logging
import threading
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from model import EvoTransformerMultiTaskV3
from genome import EvoGenomeV3
from feedback import OnlineLearner

logger = logging.getLogger("evocompliance.inference")

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
# Load Tokenizer (lightweight, instant)
# ===============================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# ===============================
# Deferred Model Initialization
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "evotransformer_v31_weights.pt")

genome = EvoGenomeV3()
model = None
learner = None
_model_ready = threading.Event()


# ===============================
# Auto-Bootstrap Training Data
# ===============================

BOOTSTRAP_TRANSACTION_DATA = [
    ("University tuition payment for fall semester", "education"),
    ("Online course subscription renewal Coursera", "education"),
    ("School textbook purchase from bookstore", "education"),
    ("Student loan repayment monthly installment", "education"),
    ("Workshop training fee professional development", "education"),
    ("College admission application fee", "education"),
    ("Tutoring service payment weekly session", "education"),
    ("Educational software license annual renewal", "education"),
    ("Netflix monthly subscription payment", "entertainment"),
    ("Movie theater tickets Saturday night", "entertainment"),
    ("Concert tickets for live music show", "entertainment"),
    ("Video game purchase from Steam store", "entertainment"),
    ("Spotify premium music streaming subscription", "entertainment"),
    ("Theme park admission tickets family outing", "entertainment"),
    ("Bowling alley weekend fun with friends", "entertainment"),
    ("Disney Plus streaming annual subscription", "entertainment"),
    ("Grocery shopping at Walmart weekly essentials", "food_grocery"),
    ("Restaurant dinner bill Italian bistro", "food_grocery"),
    ("DoorDash food delivery order lunch", "food_grocery"),
    ("Coffee shop morning latte purchase", "food_grocery"),
    ("Supermarket fresh produce and dairy items", "food_grocery"),
    ("Whole Foods organic grocery shopping", "food_grocery"),
    ("Pizza delivery Friday night dinner order", "food_grocery"),
    ("Bakery purchase fresh bread and pastries", "food_grocery"),
    ("Doctor visit copay general checkup", "healthcare"),
    ("Pharmacy prescription medication refill", "healthcare"),
    ("Dental cleaning appointment biannual visit", "healthcare"),
    ("Health insurance premium monthly payment", "healthcare"),
    ("Eye exam and new glasses purchase", "healthcare"),
    ("Hospital emergency room visit copay", "healthcare"),
    ("Physical therapy session weekly treatment", "healthcare"),
    ("Mental health counseling appointment fee", "healthcare"),
    ("Monthly salary deposit from employer", "salary_income"),
    ("Payroll direct deposit biweekly wages", "salary_income"),
    ("Freelance payment received for consulting work", "salary_income"),
    ("Bonus payment annual performance review", "salary_income"),
    ("Commission earned from sales this quarter", "salary_income"),
    ("Contractor payment for completed project", "salary_income"),
    ("Dividend income from stock portfolio", "salary_income"),
    ("Rental income monthly apartment lease", "salary_income"),
    ("Wire transfer to savings account", "transfer"),
    ("Venmo payment to friend for dinner split", "transfer"),
    ("Bank transfer between checking and savings", "transfer"),
    ("International wire transfer to family abroad", "transfer"),
    ("PayPal money transfer for shared expenses", "transfer"),
    ("Zelle payment to landlord for rent", "transfer"),
    ("ACH transfer automatic bill payment", "transfer"),
    ("Cash app transfer to roommate utilities split", "transfer"),
    ("Uber ride to airport morning commute", "transport"),
    ("Gas station fuel purchase regular unleaded", "transport"),
    ("Monthly subway pass metro transit card", "transport"),
    ("Car insurance premium quarterly payment", "transport"),
    ("Auto repair service brake replacement", "transport"),
    ("Lyft ride home from downtown bar", "transport"),
    ("Parking garage monthly pass downtown", "transport"),
    ("Flight ticket booking domestic travel", "transport"),
    ("Electric bill monthly power company payment", "utilities"),
    ("Water and sewer utility quarterly bill", "utilities"),
    ("Internet service provider monthly broadband", "utilities"),
    ("Natural gas heating bill winter month", "utilities"),
    ("Cell phone bill wireless carrier monthly", "utilities"),
    ("Trash collection service monthly fee", "utilities"),
    ("Home security monitoring monthly subscription", "utilities"),
    ("Cable television service monthly payment", "utilities"),
]

BOOTSTRAP_DOCUMENT_DATA = [
    ("The company expanded operations to three new markets in Southeast Asia during Q3", "business_operations"),
    ("Supply chain logistics were restructured to improve delivery times by 15 percent", "business_operations"),
    ("New warehouse facility opened in Dallas to support growing demand", "business_operations"),
    ("Customer service team expanded with 50 new hires across regional offices", "business_operations"),
    ("IT infrastructure upgrade completed including cloud migration of core systems", "business_operations"),
    ("Strategic partnership established with leading logistics provider for distribution", "business_operations"),
    ("Manufacturing capacity increased by 20 percent through equipment modernization", "business_operations"),
    ("Employee training program launched to upskill workforce on digital tools", "business_operations"),
    ("Revenue increased 12 percent year over year to 4.2 billion dollars", "financial_statement"),
    ("Net income for the quarter was 850 million representing a 15 percent margin", "financial_statement"),
    ("Total assets on the balance sheet amounted to 28.5 billion at year end", "financial_statement"),
    ("Operating cash flow improved to 1.8 billion from continuing operations", "financial_statement"),
    ("Earnings per share grew from 3.42 to 4.18 a 22 percent increase", "financial_statement"),
    ("Gross profit margin expanded 200 basis points to 45 percent", "financial_statement"),
    ("Long term debt decreased to 5.2 billion following scheduled repayments", "financial_statement"),
    ("Quarterly dividend of 0.75 per share declared payable to shareholders", "financial_statement"),
    ("Company is in compliance with SOX Section 404 internal control requirements", "legal_regulatory"),
    ("New GDPR privacy regulations require updated data processing agreements", "legal_regulatory"),
    ("Securities and Exchange Commission filing requirements were met on schedule", "legal_regulatory"),
    ("Anti money laundering policies updated per latest FinCEN guidance", "legal_regulatory"),
    ("Board approved updated code of ethics and business conduct policy", "legal_regulatory"),
    ("Regulatory audit completed with no material findings or deficiencies", "legal_regulatory"),
    ("Patent infringement lawsuit settled for undisclosed terms", "legal_regulatory"),
    ("Environmental compliance report submitted to EPA as required", "legal_regulatory"),
    ("The board of directors appointed Jane Smith as new Chief Executive Officer", "management_governance"),
    ("Annual general meeting approved executive compensation packages", "management_governance"),
    ("Audit committee reviewed internal controls and found no material weaknesses", "management_governance"),
    ("Board established new sustainability committee to oversee ESG initiatives", "management_governance"),
    ("Corporate governance guidelines updated to strengthen director independence", "management_governance"),
    ("Succession planning framework developed for senior leadership positions", "management_governance"),
    ("Shareholder vote approved proposed merger with competitor firm", "management_governance"),
    ("Executive team restructured with new Chief Technology Officer appointment", "management_governance"),
    ("Foreign currency fluctuations may adversely impact international revenue", "risk_disclosure"),
    ("Cybersecurity threats pose ongoing risk to customer data and operations", "risk_disclosure"),
    ("Supply chain disruptions could affect product availability and costs", "risk_disclosure"),
    ("Changes in interest rates may impact borrowing costs and profitability", "risk_disclosure"),
    ("Competitive market pressures could reduce market share and pricing power", "risk_disclosure"),
    ("Regulatory changes in key markets may require significant compliance investment", "risk_disclosure"),
    ("Climate related risks including extreme weather may disrupt operations", "risk_disclosure"),
    ("Key personnel departures could affect business continuity and performance", "risk_disclosure"),
]


def _run_bootstrap_training():
    """Train the projection layer and task heads using pretrained embeddings."""
    live_path = WEIGHTS_PATH.replace(".pt", "_live.pt")
    if os.path.exists(live_path):
        try:
            state = torch.load(live_path, map_location=DEVICE, weights_only=True)
            if "backbone.embed_projection.weight" in state:
                logger.info("Compatible live weights found — skipping bootstrap")
                return
        except Exception:
            pass

    logger.info("Running bootstrap training with pretrained embeddings...")

    model.train()

    trainable_params = [p for n, p in model.named_parameters()
                        if "token_embedding" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    all_data = (
        [(text, label, "transaction") for text, label in BOOTSTRAP_TRANSACTION_DATA]
        + [(text, label, "document") for text, label in BOOTSTRAP_DOCUMENT_DATA]
    )

    num_epochs = 50
    for epoch in range(num_epochs):
        random.shuffle(all_data)
        total_loss = 0

        for text, label, task in all_data:
            labels_list = TRANSACTION_LABELS if task == "transaction" else DOCUMENT_LABELS
            enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            target = torch.tensor([labels_list.index(label)], device=DEVICE)

            logits, _, _ = model(input_ids, attention_mask, task=task)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info("  Bootstrap epoch %d/%d | avg loss: %.4f",
                        epoch + 1, num_epochs, total_loss / len(all_data))

    model.eval()

    torch.save(model.state_dict(), WEIGHTS_PATH)
    logger.info("Bootstrap training complete — weights saved")


def _initialize_model():
    """Heavy initialization: download DistilBERT, build model, bootstrap train.

    Runs in a background thread so the server can bind its port immediately.
    """
    global model, learner

    try:
        logger.info("Loading DistilBERT pretrained embeddings...")
        distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        pretrained_embeddings = distilbert.embeddings.word_embeddings.weight.detach().clone()
        del distilbert
        logger.info("Pretrained embeddings loaded: shape %s", pretrained_embeddings.shape)

        model = EvoTransformerMultiTaskV3(
            genome, 8, 5, 9, pretrained_embeddings=pretrained_embeddings
        )

        # Load live weights if available
        live_path = WEIGHTS_PATH.replace(".pt", "_live.pt")
        if os.path.exists(live_path):
            try:
                model.load_state_dict(
                    torch.load(live_path, map_location=DEVICE, weights_only=True)
                )
                logger.info("Loaded live weights from %s", live_path)
            except RuntimeError:
                logger.warning("Live weights incompatible — will retrain")

        model.to(DEVICE)
        model.eval()

        _run_bootstrap_training()

        learner = OnlineLearner(model, genome, DEVICE, tokenizer, WEIGHTS_PATH)

        _model_ready.set()
        logger.info("Model initialization complete — ready to serve predictions")

    except Exception:
        logger.exception("Model initialization failed")


# Start initialization in background thread
_init_thread = threading.Thread(target=_initialize_model, daemon=True)
_init_thread.start()


# ===============================
# Prediction Function
# ===============================

def predict(text, task):
    if not _model_ready.is_set():
        return {
            "error": "Model is warming up. Please retry in a few seconds.",
            "status": "initializing"
        }

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
