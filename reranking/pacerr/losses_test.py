import torch
from typing import Union, Tuple, List, Iterable, Dict
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import MSELoss
"""
Groupwuse Hinge Loss V1: 
- Dilated hinge loss with stride 
    # [cfda] move into loss.py

"""
class GroupwiseMSELoss(nn.Module):

    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MSELoss(reduction=reduction)
        self.activation = nn.Sigmoid()

    def forward(self, logits: Tensor, labels: Tensor):
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        labels = labels.view(-1, self.examples_per_group)
        return self.loss_fct(logits, labels)

