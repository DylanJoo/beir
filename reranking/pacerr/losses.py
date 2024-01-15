import math
import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from torch.nn import functional as F
from torch.nn import MarginRankingLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

class PairwiseHingeLoss(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = MarginRankingLoss(margin=margin, reduction=reduction)
        self.activation = nn.Sigmoid()

    def forward(self, logits: Tensor, labels: Tensor):
        """ Try using labels as filter"""
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        logits_negaitve = logits[:, 0] # see `filter`
        logits_positive = logits[:, 1] # see `filter`
        targets = torch.ones(logits.size(0)).to(logits.device) 
        # 1 means left>right
        loss = self.loss_fct(logits_positive, logits_negaitve, targets)
        return loss

class GroupwiseHingeLoss(PairwiseHingeLoss):
    """
    n = N-1 (num examples - 1)
    - \sum_{i=1}^n PairwiseHingeLoss( [q0, p], [qi, p] )
    [NOTE 1] It can also be like warp, 
    """
    def forward(self, logits: Tensor, labels: Tensor):
        loss = 0
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.ones(logits.size(0)).to(logits.device)
        for i in range(logits.size(-1)-1):
            loss += self.loss_fct(logits[:, 0], logits[:, i+1], targets)
        return loss / (logits.size(-1) - 1)

class GroupwiseHingeLossV1(nn.Module):
    """
    - \sum_{i=0}^n HingeLoss( [qi, p], [qi+1, p] ) # dilated 
    - \sum_{i=0}^n \sum_{j=i+1}^n HingeLoss( [qi, p], [qj, p] )
    """
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
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

class CELoss(nn.Module):
    """
    The original LCELoss is not pairwise. It's only a special case.
    - Temperature is a hyperparameter.

    If n = 2, the pairwise CE Loss
    - LCELoss( [[q0, p], [q1, p], ...[qn, p]] )

    If n > 2, the groupwise CE Loss
    - \sum_{i=0}^n CELoss( [[qi, p], [qi+1, p], ...[qn, p]] )
    """
    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction: str = 'mean', 
                 batchwise: bool = False):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = CrossEntropyLoss(reduction=reduction)
        self.batchwise = batchwise

    def forward(self, logits: Tensor, labels: Tensor):
        n = self.examples_per_group
        if self.batchwise:
            n = int(math.sqrt( logits.size(0) // n )) # batch_d #q x batch_q
        logits = logits.view(-1, n) # reshape (B 1)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        return self.loss_fct(logits, targets)

class GroupwiseCELossV1(nn.Module):
    """
    - \sum_{i=0}^n HingeLoss( [qi, p], [qi+1, p] ) # dilated 
    - \sum_{i=0}^n \sum_{j=i+1}^n HingeLoss( [qi, p], [qj, p] )
    """
    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
                 stride: int = 1,    # the size between selected positions
                 dilation: int = 1,  # the position of the paired negative 
                 reduction: str = 'mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = CrossEntropyLoss(reduction=reduction)
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
        logits = logits.view(-1, self.examples_per_group)
        for idx in self.sample_indices:
            logits_positive = logits[:, idx]
            logits_negative = logits[:, (idx+self.dilation)]
            loss += self.loss_fct(logits_positive, logits_negative)
        return loss / len(self.sample_indices)

class MSELoss(nn.Module):

    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction='mean'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.activation = nn.Sigmoid()
        self.loss_fct = MSELoss(reduction='mean')

    def forward(self, logits: Tensor, labels: Tensor):
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        labels = labels.view(-1, self.examples_per_group)
        return self.loss_fct(logits, labels)

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

