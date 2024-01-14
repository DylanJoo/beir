import torch
from typing import Union, Tuple, List, Iterable, Dict
from torch import nn, Tensor
from torch.nn import functional as F
"""
Groupwuse Hinge Loss V1: 
- Dilated hinge loss with stride 
"""

class GroupwiseHingeLossV1(nn.Module):

    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 0, 
                 stride: int = 1,    # the size between selected positions
                 dilation: int = 1,  # the position of the paired negative 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MarginRankingLoss(
                margin=margin, 
                reduction=reduction
        )
        self.activation = nn.Sigmoid()
        self.stride = stride
        self.dilation = dilation
        self.sample_indices = list(
                range(0, examples_per_group-dilation, stride)
        )

        for i, idx in enumerate(self.sample_indices):
            print(f"The {i+1} pair: + {idx}; - {idx+dilation}")

    def forward(self, logits, labels):
        loss = 0
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.ones(logits.size(0)).to(logits.device)
        for idx in self.sample_indices:
            logits_positive = logits[:, idx]
            logits_negative = logits[:, (idx+self.dilation)]
            loss += self.loss_fct(logits_positive, logits_negative)
        return loss / len(self.sample_indices)

