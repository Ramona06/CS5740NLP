# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: KL873

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ner.nn.module import Module


class FFNN(Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1) -> None:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        super().__init__()

        assert num_layers > 0

        # TODO-4-1
        self.W = nn.Linear(embedding_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, output_dim)

        self.U = nn.ModuleList()
        for i in range(num_layers - 1):
          self.U.append(nn.Linear(hidden_dim, hidden_dim))

        self.V = nn.Linear(hidden_dim, output_dim)
        
        self.apply(self.init_weights)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.ffnn.html."""
        # TODO-4-2
        # W_trans = torch.transpose(self.W(embeddings))
        Z_prime1 = self.W(embeddings)
        Z1 = F.relu(Z_prime1)

        # Initialize Z for the first hidden layer
        Z = Z1
        # For more layers
        for layer in self.U:
            Z_prime_k = layer(Z)
            Z_k = F.relu(Z_prime_k)
            Z = Z_k  # Update Z for the next layer

        # Use V to transform the last hidden intermediate Zn to Y'
        Y_prime = self.V(Z)
        return Y_prime
        #raise NotImplementedError  # remove once the method is filled
