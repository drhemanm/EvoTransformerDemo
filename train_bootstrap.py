"""
Bootstrap training script for improving model accuracy with synthetic data.

NOTE: The current EvoTransformer architecture uses random token embeddings
(not pretrained). For production-grade accuracy, swap in pretrained embeddings
from DistilBERT or fine-tune on a large labeled dataset (1000+ samples per class).

The online learning (feedback) loop is the key differentiator — accuracy improves
incrementally as users correct predictions in production.

Usage:
    python train_bootstrap.py
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
from model import EvoTransformerMultiTaskV3
from genome import EvoGenomeV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Synthetic Training Data
# ===============================

TRANSACTION_DATA = [
    # education
    ("University tuition payment for fall semester", "education"),
    ("Online course subscription renewal Coursera", "education"),
    ("School textbook purchase from bookstore", "education"),
    ("Student loan repayment monthly installment", "education"),
    ("Workshop training fee professional development", "education"),
    ("College admission application fee", "education"),
    ("Tutoring service payment weekly session", "education"),
    ("Educational software license annual renewal", "education"),
    # entertainment
    ("Netflix monthly subscription payment", "entertainment"),
    ("Movie theater tickets Saturday night", "entertainment"),
    ("Concert tickets for live music show", "entertainment"),
    ("Video game purchase from Steam store", "entertainment"),
    ("Spotify premium music streaming subscription", "entertainment"),
    ("Theme park admission tickets family outing", "entertainment"),
    ("Bowling alley weekend fun with friends", "entertainment"),
    ("Disney Plus streaming annual subscription", "entertainment"),
    # food_grocery
    ("Grocery shopping at Walmart weekly essentials", "food_grocery"),
    ("Restaurant dinner bill Italian bistro", "food_grocery"),
    ("DoorDash food delivery order lunch", "food_grocery"),
    ("Coffee shop morning latte purchase", "food_grocery"),
    ("Supermarket fresh produce and dairy items", "food_grocery"),
    ("Whole Foods organic grocery shopping", "food_grocery"),
    ("Pizza delivery Friday night dinner order", "food_grocery"),
    ("Bakery purchase fresh bread and pastries", "food_grocery"),
    # healthcare
    ("Doctor visit copay general checkup", "healthcare"),
    ("Pharmacy prescription medication refill", "healthcare"),
    ("Dental cleaning appointment biannual visit", "healthcare"),
    ("Health insurance premium monthly payment", "healthcare"),
    ("Eye exam and new glasses purchase", "healthcare"),
    ("Hospital emergency room visit copay", "healthcare"),
    ("Physical therapy session weekly treatment", "healthcare"),
    ("Mental health counseling appointment fee", "healthcare"),
    # salary_income
    ("Monthly salary deposit from employer", "salary_income"),
    ("Payroll direct deposit biweekly wages", "salary_income"),
    ("Freelance payment received for consulting work", "salary_income"),
    ("Bonus payment annual performance review", "salary_income"),
    ("Commission earned from sales this quarter", "salary_income"),
    ("Contractor payment for completed project", "salary_income"),
    ("Dividend income from stock portfolio", "salary_income"),
    ("Rental income monthly apartment lease", "salary_income"),
    # transfer
    ("Wire transfer to savings account", "transfer"),
    ("Venmo payment to friend for dinner split", "transfer"),
    ("Bank transfer between checking and savings", "transfer"),
    ("International wire transfer to family abroad", "transfer"),
    ("PayPal money transfer for shared expenses", "transfer"),
    ("Zelle payment to landlord for rent", "transfer"),
    ("ACH transfer automatic bill payment", "transfer"),
    ("Cash app transfer to roommate utilities split", "transfer"),
    # transport
    ("Uber ride to airport morning commute", "transport"),
    ("Gas station fuel purchase regular unleaded", "transport"),
    ("Monthly subway pass metro transit card", "transport"),
    ("Car insurance premium quarterly payment", "transport"),
    ("Auto repair service brake replacement", "transport"),
    ("Lyft ride home from downtown bar", "transport"),
    ("Parking garage monthly pass downtown", "transport"),
    ("Flight ticket booking domestic travel", "transport"),
    # utilities
    ("Electric bill monthly power company payment", "utilities"),
    ("Water and sewer utility quarterly bill", "utilities"),
    ("Internet service provider monthly broadband", "utilities"),
    ("Natural gas heating bill winter month", "utilities"),
    ("Cell phone bill wireless carrier monthly", "utilities"),
    ("Trash collection service monthly fee", "utilities"),
    ("Home security monitoring monthly subscription", "utilities"),
    ("Cable television service monthly payment", "utilities"),
]

DOCUMENT_DATA = [
    # business_operations
    ("The company expanded operations to three new markets in Southeast Asia during Q3", "business_operations"),
    ("Supply chain logistics were restructured to improve delivery times by 15 percent", "business_operations"),
    ("New warehouse facility opened in Dallas to support growing demand", "business_operations"),
    ("Customer service team expanded with 50 new hires across regional offices", "business_operations"),
    ("IT infrastructure upgrade completed including cloud migration of core systems", "business_operations"),
    ("Strategic partnership established with leading logistics provider for distribution", "business_operations"),
    ("Manufacturing capacity increased by 20 percent through equipment modernization", "business_operations"),
    ("Employee training program launched to upskill workforce on digital tools", "business_operations"),
    # financial_statement
    ("Revenue increased 12 percent year over year to 4.2 billion dollars", "financial_statement"),
    ("Net income for the quarter was 850 million representing a 15 percent margin", "financial_statement"),
    ("Total assets on the balance sheet amounted to 28.5 billion at year end", "financial_statement"),
    ("Operating cash flow improved to 1.8 billion from continuing operations", "financial_statement"),
    ("Earnings per share grew from 3.42 to 4.18 a 22 percent increase", "financial_statement"),
    ("Gross profit margin expanded 200 basis points to 45 percent", "financial_statement"),
    ("Long term debt decreased to 5.2 billion following scheduled repayments", "financial_statement"),
    ("Quarterly dividend of 0.75 per share declared payable to shareholders", "financial_statement"),
    # legal_regulatory
    ("Company is in compliance with SOX Section 404 internal control requirements", "legal_regulatory"),
    ("New GDPR privacy regulations require updated data processing agreements", "legal_regulatory"),
    ("Securities and Exchange Commission filing requirements were met on schedule", "legal_regulatory"),
    ("Anti money laundering policies updated per latest FinCEN guidance", "legal_regulatory"),
    ("Board approved updated code of ethics and business conduct policy", "legal_regulatory"),
    ("Regulatory audit completed with no material findings or deficiencies", "legal_regulatory"),
    ("Patent infringement lawsuit settled for undisclosed terms", "legal_regulatory"),
    ("Environmental compliance report submitted to EPA as required", "legal_regulatory"),
    # management_governance
    ("The board of directors appointed Jane Smith as new Chief Executive Officer", "management_governance"),
    ("Annual general meeting approved executive compensation packages", "management_governance"),
    ("Audit committee reviewed internal controls and found no material weaknesses", "management_governance"),
    ("Board established new sustainability committee to oversee ESG initiatives", "management_governance"),
    ("Corporate governance guidelines updated to strengthen director independence", "management_governance"),
    ("Succession planning framework developed for senior leadership positions", "management_governance"),
    ("Shareholder vote approved proposed merger with competitor firm", "management_governance"),
    ("Executive team restructured with new Chief Technology Officer appointment", "management_governance"),
    # risk_disclosure
    ("Foreign currency fluctuations may adversely impact international revenue", "risk_disclosure"),
    ("Cybersecurity threats pose ongoing risk to customer data and operations", "risk_disclosure"),
    ("Supply chain disruptions could affect product availability and costs", "risk_disclosure"),
    ("Changes in interest rates may impact borrowing costs and profitability", "risk_disclosure"),
    ("Competitive market pressures could reduce market share and pricing power", "risk_disclosure"),
    ("Regulatory changes in key markets may require significant compliance investment", "risk_disclosure"),
    ("Climate related risks including extreme weather may disrupt operations", "risk_disclosure"),
    ("Key personnel departures could affect business continuity and performance", "risk_disclosure"),
]

TRANSACTION_LABELS = [
    "education", "entertainment", "food_grocery", "healthcare",
    "salary_income", "transfer", "transport", "utilities"
]

DOCUMENT_LABELS = [
    "business_operations", "financial_statement",
    "legal_regulatory", "management_governance", "risk_disclosure"
]


def train():
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    print("Building model...")
    genome = EvoGenomeV3()
    model = EvoTransformerMultiTaskV3(genome, 8, 5, 9)

    # Load existing weights
    weights_path = "evotransformer_v31_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    NUM_EPOCHS = 300

    # Interleaved multi-task training
    print(f"\nTraining both tasks interleaved ({NUM_EPOCHS} epochs)...")
    all_data = [(text, label, "transaction") for text, label in TRANSACTION_DATA] + \
               [(text, label, "document") for text, label in DOCUMENT_DATA]

    for epoch in range(NUM_EPOCHS):
        import random
        random.shuffle(all_data)

        total_loss = 0
        txn_correct = txn_total = 0
        doc_correct = doc_total = 0

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
            pred = logits.argmax(-1).item()
            if task == "transaction":
                txn_correct += (pred == target.item())
                txn_total += 1
            else:
                doc_correct += (pred == target.item())
                doc_total += 1

        if (epoch + 1) % 10 == 0:
            txn_acc = txn_correct / txn_total * 100 if txn_total else 0
            doc_acc = doc_correct / doc_total * 100 if doc_total else 0
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {total_loss/len(all_data):.4f} | Txn: {txn_acc:.1f}% | Doc: {doc_acc:.1f}%")

    # Evaluate
    model.eval()
    print("\n--- Final Evaluation ---")

    for task_name, data, labels in [
        ("transaction", TRANSACTION_DATA, TRANSACTION_LABELS),
        ("document", DOCUMENT_DATA, DOCUMENT_LABELS)
    ]:
        correct = 0
        total = 0
        for text, label in data:
            enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad():
                logits, _, _ = model(input_ids, attention_mask, task=task_name)
            pred = logits.argmax(-1).item()
            correct += (pred == labels.index(label))
            total += 1
        print(f"  {task_name}: {correct}/{total} = {correct/total*100:.1f}%")

    # Save updated weights
    print("\nSaving updated weights...")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved to {weights_path}")
    print("Done!")


if __name__ == "__main__":
    train()
