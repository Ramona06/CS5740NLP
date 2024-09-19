# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: KL873

import logging
from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from ner.nn.module import Module


class RNN(Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        super().__init__()

        assert num_layers > 0

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        logging.info(f"no shared weights across layers")

        nonlinearity_dict = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "prelu": nn.PReLU()}
        if nonlinearity not in nonlinearity_dict:
            raise ValueError(f"{nonlinearity} not supported, choose one of: [tanh, relu, prelu]")
        self.nonlinear = nonlinearity_dict[nonlinearity]

        # TODO-5-1
        self.W = nn.ModuleList()
        for i in range(num_layers):
          if i==0:
            self.W.append(nn.Linear(embedding_dim, hidden_dim, bias=bias))
          else:
            self.W.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))

        self.U = nn.ModuleList()
        for i in range(num_layers):
          self.U.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
          
        self.V = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.apply(self.init_weights)

    def _initial_hidden_states(
        self, batch_size: int, init_zeros: bool = False, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        if init_zeros:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        else:
            hidden_states = nn.init.xavier_normal_(
                torch.empty(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        return list(map(torch.squeeze, hidden_states))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.nn.models.rnn.html."""
        # TODO-5-2
        batch_size, batch_max_length, _ = embeddings.shape

        # Initialize hidden states
        hidden_states = self._initial_hidden_states(batch_size, device=embeddings.device)

        #print(batch_max_length, batch_size)
        # # Forward pass through time dimension
        yt = []
        for t in range(batch_max_length):
          xt = embeddings[:, t, :]  # Input at time step t
          #print("xt", xt)
          #print(torch.sum(xt))
          for k in range(self.num_layers):
              if k == 0:
                  # For the first layer (k=0)
                  hidden_states[k] = self.nonlinear(self.W[k](xt) + self.U[k](hidden_states[k]))
              else:
                  # For subsequent layers (k>0)
                  hidden_states[k] = self.nonlinear(self.W[k](hidden_states[k-1]) + self.U[k](hidden_states[k]))
          yt.append(self.V(hidden_states[k]))
        
        #stack for each time step
        yt_stacked = torch.stack(yt, dim = 1) #dimension is (batch_size, time_step, 9)

        #Single-Layer RNN
        # for t in range(batch_max_length):
        #   xt = embeddings[:, t, :]  # Input at time step t
        #   zt = self.nonlinear(self.W(xt) + self.U(hidden_states[-1]))
        #   yt = self.V(zt)
        #   hidden_states.append(zt) update the hidden_states

        return yt_stacked
