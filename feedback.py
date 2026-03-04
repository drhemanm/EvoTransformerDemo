import os
import copy
import threading
import torch
import torch.nn as nn
from collections import deque
from transformers import DistilBertTokenizer


# ===============================
# Feedback Store
# ===============================

class FeedbackBuffer:
    """Thread-safe circular buffer for storing user feedback."""

    def __init__(self, max_size=256):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, text, task, correct_label):
        with self.lock:
            self.buffer.append({
                "text": text,
                "task": task,
                "correct_label": correct_label
            })

    def get_batch(self, batch_size):
        with self.lock:
            items = list(self.buffer)[:batch_size]
        return items

    def drain(self, count):
        """Remove the oldest `count` items after they've been learned from."""
        with self.lock:
            for _ in range(min(count, len(self.buffer))):
                self.buffer.popleft()

    def size(self):
        with self.lock:
            return len(self.buffer)


# ===============================
# Label Lookups (reuse from inference)
# ===============================

TASK_LABELS = {
    "transaction": [
        "education", "entertainment", "food_grocery", "healthcare",
        "salary_income", "transfer", "transport", "utilities"
    ],
    "document": [
        "business_operations", "financial_statement",
        "legal_regulatory", "management_governance", "risk_disclosure"
    ],
    "ner": [
        "O", "B-ORG", "I-ORG", "B-PER", "I-PER",
        "B-LOC", "I-LOC", "B-MONEY", "I-MONEY"
    ]
}


def label_to_index(task, label_str):
    """Convert a label string to its integer index for the given task."""
    labels = TASK_LABELS.get(task)
    if labels is None:
        return None
    try:
        return labels.index(label_str)
    except ValueError:
        return None


# ===============================
# Online Learner
# ===============================

class OnlineLearner:
    """Handles live micro-training from user feedback without full retraining."""

    def __init__(self, model, genome, device, tokenizer, weights_path):
        self.model = model
        self.genome = genome
        self.device = device
        self.tokenizer = tokenizer
        self.weights_path = weights_path
        self.buffer = FeedbackBuffer(max_size=genome.max_feedback_buffer)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=genome.online_lr,
            weight_decay=0.01
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.lock = threading.Lock()
        self.learn_count = 0

    def add_feedback(self, text, task, correct_label):
        """Store a feedback sample. Returns True if learning was triggered."""
        idx = label_to_index(task, correct_label)
        if idx is None:
            return False, "Invalid label for task"

        self.buffer.add(text, task, correct_label)

        if self.buffer.size() >= self.genome.feedback_batch_size:
            self._micro_train()
            return True, "Feedback recorded and model updated"

        return True, "Feedback recorded (buffered for next update)"

    def _micro_train(self):
        """Run a single micro-training step on buffered feedback."""
        batch = self.buffer.get_batch(self.genome.feedback_batch_size)
        if not batch:
            return

        with self.lock:
            self.model.train()

            for sample in batch:
                text = sample["text"]
                task = sample["task"]
                label_idx = label_to_index(task, sample["correct_label"])

                if label_idx is None:
                    continue

                enc = self.tokenizer(
                    text,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                if task in ("transaction", "document"):
                    target = torch.tensor([label_idx], device=self.device)
                else:
                    # NER: apply label to all non-special tokens
                    seq_len = input_ids.shape[1]
                    target = torch.full((1, seq_len), label_idx,
                                        dtype=torch.long, device=self.device)

                logits, _, _ = self.model(input_ids, attention_mask, task=task)

                if task == "ner":
                    logits = logits.view(-1, logits.shape[-1])
                    target = target.view(-1)

                loss = self.loss_fn(logits, target)

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent catastrophic updates
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

            self.model.eval()
            self.learn_count += len(batch)
            self.buffer.drain(len(batch))

            # Checkpoint updated weights
            self._save_checkpoint()

    def _save_checkpoint(self):
        """Save updated weights to disk."""
        checkpoint_path = self.weights_path.replace(".pt", "_live.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def get_stats(self):
        return {
            "feedback_buffered": self.buffer.size(),
            "total_learned": self.learn_count,
            "learning_rate": self.genome.online_lr,
            "batch_size": self.genome.feedback_batch_size
        }
