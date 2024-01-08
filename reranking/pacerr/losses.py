import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from torch.nn import functional as F
from torch.nn import MarginRankingLoss
from torch.nn import BCEWithLogitsLoss

class CombinedLoss(nn.Module):

    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 0,
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct_0 = \
                PairwiseHingeLoss(examples_per_group, margin, reduction)
        self.loss_fct_1 = BCEWithLogitsLoss()

    def forward(self, logits: Tensor, labels: Tensor):
        loss_groupwise = self.loss_fct_0(logits, labels)
        loss_pointwise = self.loss_fct_1(logits, labels)
        loss = loss_groupwise + loss_pointwise
        return loss_groupwise

class PairwiseHingeLoss(nn.Module):
    """
    Compute the loss between two query-document pairs with margin.
    """
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, logits: Tensor, labels: Tensor):
        # reshape the logits (B 1)
        logits = logits.view(-1, self.examples_per_group)
        # left should large than right
        logits_negaitve = logits[:, 0] # see `filter`
        logits_positive = logits[:, 1] # see `filter`
        targets = torch.ones(logits.size(0)).to(logits.device)
        return self.loss_fct(logits_positive, logits_negaitve, targets)

