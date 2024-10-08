# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: KL873

from dataclasses import dataclass

import torch
from torch import nn

from ner.nn.module import Module
import torch.nn.functional as F


class TokenEmbedding(Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.embeddings.embedding.html."""
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.apply(self.init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.embeddings.embedding.html."""
        # TODO-2-2
        embedded_tokens = self.embedding(input_ids)
        
        return embedded_tokens
        
