"""Tests for the EvoTransformer model architecture."""

import torch
import pytest
from model import EvoTransformerLayerV3, EvoTransformerBackboneV3, EvoTransformerMultiTaskV3
from genome import EvoGenomeV3


@pytest.fixture
def genome():
    return EvoGenomeV3()


@pytest.fixture
def model(genome):
    return EvoTransformerMultiTaskV3(genome, 8, 5, 9)


def test_genome_defaults():
    """Genome should have expected default values."""
    g = EvoGenomeV3()
    assert g.num_layers == 2
    assert g.embed_dim == 128
    assert g.ffn_dim == 512
    assert g.use_early_exit is True


def test_model_creation(model):
    """Model should be created without errors."""
    assert model is not None
    assert hasattr(model, "backbone")
    assert hasattr(model, "transaction_head")
    assert hasattr(model, "document_head")
    assert hasattr(model, "ner_head")


def test_model_forward_transaction(model):
    """Transaction forward pass should produce correct output shape."""
    input_ids = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    logits, active_layers, exit_confs = model(input_ids, attention_mask, task="transaction")
    assert logits.shape == (1, 8)  # 8 transaction labels
    assert active_layers > 0


def test_model_forward_document(model):
    """Document forward pass should produce correct output shape."""
    input_ids = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    logits, active_layers, exit_confs = model(input_ids, attention_mask, task="document")
    assert logits.shape == (1, 5)  # 5 document labels


def test_model_forward_ner(model):
    """NER forward pass should produce correct output shape."""
    input_ids = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    logits, active_layers, exit_confs = model(input_ids, attention_mask, task="ner")
    assert logits.shape == (1, 128, 9)  # 9 NER labels per token


def test_model_invalid_task(model):
    """Invalid task should raise ValueError."""
    input_ids = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids, attention_mask, task="invalid")


def test_model_batch(model):
    """Model should handle batch sizes > 1."""
    input_ids = torch.randint(0, 30522, (4, 128))
    attention_mask = torch.ones(4, 128, dtype=torch.long)

    logits, _, _ = model(input_ids, attention_mask, task="transaction")
    assert logits.shape == (4, 8)


def test_early_exit(genome, model):
    """Early exit should be enabled when configured."""
    assert genome.use_early_exit is True
    for layer in model.backbone.layers:
        assert layer.exit_gate is not None


def test_model_with_pretrained_embeddings():
    """Model should work with pretrained embeddings and projection layer."""
    genome = EvoGenomeV3()
    # Simulate pretrained embeddings (768-dim like DistilBERT)
    fake_embeddings = torch.randn(30522, 768)
    model = EvoTransformerMultiTaskV3(
        genome, 8, 5, 9, pretrained_embeddings=fake_embeddings
    )
    assert model.backbone.embed_projection is not None

    input_ids = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)
    logits, _, _ = model(input_ids, attention_mask, task="transaction")
    assert logits.shape == (1, 8)


def test_model_without_pretrained_embeddings():
    """Model should still work without pretrained embeddings (random init)."""
    genome = EvoGenomeV3()
    model = EvoTransformerMultiTaskV3(genome, 8, 5, 9)
    assert model.backbone.embed_projection is None

    input_ids = torch.randint(0, 30522, (1, 128))
    attention_mask = torch.ones(1, 128, dtype=torch.long)
    logits, _, _ = model(input_ids, attention_mask, task="transaction")
    assert logits.shape == (1, 8)
