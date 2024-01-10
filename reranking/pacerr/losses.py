import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from torch.nn import functional as F
from torch.nn import MarginRankingLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss

class PairwiseHingeLoss(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, logits: Tensor, labels: Tensor):
        """ Try using labels as filter"""
        logits = logits.view(-1, self.examples_per_group)
        logits_negaitve = logits[:, 0] # see `filter`
        logits_positive = logits[:, 1] # see `filter`
        targets = torch.zeros(logits.size(0)).to(logits.device)
        loss = self.loss_fct(logits_positive, logits_negaitve, targets)
        return loss

# actually it can support multiple negatives
class PairwiseLCELoss(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: Tensor, labels: Tensor):
        """ Try using labels as filter"""
        logits = logits.view(-1, self.examples_per_group) # reshape (B 1)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        return self.loss_fct(logits, targets)

class GroupwiseHingeLoss(PairwiseHingeLoss):
    """
    n = N-1 (num examples - 1)

    - \sum_{i=1}^n PairwiseHingeLoss( [q0, p], [qi, p] )
    [NOTE 1] It can also be like warp, which mean select the n-th negative that have loss

    - \sum_{i=0}^n PairwiseHingeLoss( [qi, p], [qi+1, p] )
    - \sum_{i=0}^n \sum_{j=i+1}^n PairwiseHingeLoss( [qi, p], [qj, p] ): this combines the above methods.
    """
    def forward(self, logits: Tensor, labels: Tensor):
        loss = 0
        logits = logits.view(-1, self.examples_per_group)
        logits_positive = logits[:, 0]
        targets = torch.zeros(logits.size(0)).to(logits.device)
        for i in range(logits.size(-1)-1):
            loss += self.loss_fct(logits[:, 0], logits[:, i+1], targets)
        return loss / (logits.size(-1) - 1)

class GroupwiseLCELoss(PairwiseLCELoss):
    """
    The original LCELoss is not pairwise. It's only a special case.
    n = N-1 (num examples - 1) 

    - Temperature is a hyperparameter.

    - LCELoss( [[q0, p], [q1, p], ...[qn, p]] )
    - \sum_{i=0}^n LCELoss( [[qi, p], [qi+1, p], ...[qn, p]] )
    """
    def forward(self, logits: Tensor, labels: Tensor):
        loss = 0
        logits = logits.view(-1, self.examples_per_group) # reshape (B 1)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        return self.loss_fct(logits, targets)

class CombinedLoss(nn.Module):
    def __init__(self, 
                 add_hinge_loss: bool = False,
                 add_lce_loss: bool = False,
                 examples_per_group: int = 1, 
                 margin: float = 0,
                 reduction: str = 'mean'):

        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = [BCEWithLogitsLoss()]

        if add_hinge_loss:
            self.loss_fct += [
                    PairwiseHingeLoss(examples_per_group, margin, reduction)
            ]
        if add_lce_loss:
            self.loss_fct += [
                    PairwiseLCELoss(examples_per_group, reduction)
            ]

    def forward(self, logits: Tensor, labels: Tensor):
        loss = 0
        for loss_fct in self.loss_fct:
            loss += loss_fct(logits, labels)
        return loss

