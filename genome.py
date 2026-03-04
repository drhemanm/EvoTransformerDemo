from dataclasses import dataclass
from typing import Tuple

@dataclass
class EvoGenomeV3:
    num_layers: int = 2
    embed_dim: int = 128
    ffn_dim: int = 512
    dropout: float = 0.05
    activation: str = "relu"
    pool_strategy: str = "cls"
    attention_types: Tuple[str, ...] = ("full", "full")
    heads_per_layer: Tuple[int, ...] = (8, 1)
    connectivity: str = "reuse_first"
    use_early_exit: bool = True
    early_exit_threshold: float = 0.50
    use_layernorm: bool = False
    # Online learning config
    online_lr: float = 1e-4
    feedback_batch_size: int = 8
    max_feedback_buffer: int = 256
    confidence_threshold: float = 0.85
