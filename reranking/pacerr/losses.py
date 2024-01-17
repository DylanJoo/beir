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
    """ [NOTE 1] It can also be like warp """
    def forward(self, logits: Tensor, labels: Tensor):
        loss = 0
        logits = self.activation(logits)
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.ones(logits.size(0)).to(logits.device)
        for idx in range(logits.size(-1)-1):
            loss += self.loss_fct(logits[:, 0], logits[:, idx+1], targets)
        return loss / (logits.size(-1) - 1)

class GroupwiseHingeLossV1(nn.Module):
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
            logits_negative = logits[:, (idx+self.dilation)]
            loss += self.loss_fct(logits[:, idx], logits_negative, targets)
        return loss / len(self.sample_indices)

class CELoss(nn.Module):
    """ [NOTE] Temperature is a hyperparameter. """
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

class GroupwiseCELoss(nn.Module):
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

    def forward(self, logits, labels):
        loss = 0
        logits = logits.view(-1, self.examples_per_group)
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        for idx in range(logits.size(-1)-1):
            loss += self.loss_fct(logits[:, [0, idx+1]], targets)
        return loss / (logits.size(-1) - 1)

class GroupwiseCELossV1(nn.Module):
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
        targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        for idx in self.sample_indices:
            logits_ = logits[:, [idx, (idx+self.dilation)] ]
            loss += self.loss_fct(logits_, targets)
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
