import torch
from transformers import DistilBertTokenizer
from model import EvoTransformerMultiTaskV3
from genome import EvoGenomeV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

genome = EvoGenomeV3()
model = EvoTransformerMultiTaskV3(genome, 8, 5, 9)
model.load_state_dict(torch.load("evotransformer_v31_weights.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(text, task):
    enc = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits, active_layers, _ = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE),
            task=task
        )

    if task == "ner":
        preds = logits.argmax(-1).cpu().tolist()
        return {"predictions": preds, "active_layers": active_layers}
    else:
        pred = logits.argmax(-1).item()
        return {"prediction": pred, "active_layers": active_layers}
