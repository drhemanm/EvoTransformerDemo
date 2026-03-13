import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# EvoTransformer Layer
# ================================

class EvoTransformerLayerV3(nn.Module):
    def __init__(self, genome, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx
        embed_dim = genome.embed_dim
        num_heads = genome.heads_per_layer[layer_idx]

        # Full attention only (your winning genome uses full/full)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=genome.dropout,
            batch_first=True
        )

        # Feedforward
        activation = nn.ReLU() if genome.activation == "relu" else nn.GELU()

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, genome.ffn_dim),
            activation,
            nn.Dropout(genome.dropout),
            nn.Linear(genome.ffn_dim, embed_dim),
            nn.Dropout(genome.dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim) if genome.use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if genome.use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(genome.dropout)

        # Early exit gate
        if genome.use_early_exit:
            self.exit_gate = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.exit_gate = None

    def forward(self, x, attention_mask=None):
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))

        exit_conf = None
        if self.exit_gate is not None:
            exit_conf = self.exit_gate(x[:, 0]).squeeze(-1)

        return x, exit_conf


# ================================
# Backbone
# ================================

class EvoTransformerBackboneV3(nn.Module):
    def __init__(self, genome, vocab_size=30522, max_seq_len=128, pretrained_embeddings=None):
        super().__init__()

        self.genome = genome

        if pretrained_embeddings is not None:
            pretrained_dim = pretrained_embeddings.shape[1]
            self.token_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=True
            )
            self.embed_projection = nn.Linear(pretrained_dim, genome.embed_dim)
        else:
            self.token_embedding = nn.Embedding(vocab_size, genome.embed_dim)
            self.embed_projection = None

        self.position_embedding = nn.Embedding(max_seq_len, genome.embed_dim)

        self.embedding_dropout = nn.Dropout(genome.dropout)
        self.embedding_norm = nn.LayerNorm(genome.embed_dim) if genome.use_layernorm else nn.Identity()

        self.layers = nn.ModuleList([
            EvoTransformerLayerV3(genome, i)
            for i in range(genome.num_layers)
        ])

        self.pool_strategy = genome.pool_strategy

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape

        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids)
        if self.embed_projection is not None:
            x = self.embed_projection(x)
        x = x + self.position_embedding(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        active_layers = 0
        exit_confs = []

        for i, layer in enumerate(self.layers):
            x, exit_conf = layer(x, attention_mask)
            active_layers += 1

            if exit_conf is not None:
                exit_confs.append(exit_conf)
                if not self.training and exit_conf.mean() > self.genome.early_exit_threshold:
                    break

        if self.pool_strategy == "cls":
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)

        return x, pooled, active_layers, exit_confs


# ================================
# Multi-Task Head
# ================================

class EvoTransformerMultiTaskV3(nn.Module):
    def __init__(self, genome, num_txn_labels, num_doc_labels, num_ner_labels,
                 pretrained_embeddings=None):
        super().__init__()

        self.genome = genome
        self.backbone = EvoTransformerBackboneV3(
            genome, pretrained_embeddings=pretrained_embeddings
        )

        self.transaction_head = nn.Sequential(
            nn.Linear(genome.embed_dim, genome.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(genome.dropout),
            nn.Linear(genome.embed_dim // 2, num_txn_labels)
        )

        self.document_head = nn.Sequential(
            nn.Linear(genome.embed_dim, genome.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(genome.dropout),
            nn.Linear(genome.embed_dim // 2, num_doc_labels)
        )

        self.ner_head = nn.Sequential(
            nn.Linear(genome.embed_dim, genome.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(genome.dropout),
            nn.Linear(genome.embed_dim // 2, num_ner_labels)
        )

    def forward(self, input_ids, attention_mask=None, task="transaction"):
        seq_out, pooled, active_layers, exit_confs = self.backbone(input_ids, attention_mask)

        if task == "transaction":
            logits = self.transaction_head(pooled)
        elif task == "document":
            logits = self.document_head(pooled)
        elif task == "ner":
            logits = self.ner_head(seq_out)
        else:
            raise ValueError("Invalid task")

        return logits, active_layers, exit_confs
