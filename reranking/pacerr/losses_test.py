import torch
from typing import Union, Tuple, List, Iterable, Dict
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

class LocalizedIBCELoss(nn.Module):

    def __init__(self, 
                 examples_per_group: int = 1, 
                 margin: float = 1, 
                 stride: int = 1,    # the size between selected positions
                 dilation: int = 1,  # the position of the paired negative 
                 reduction: str = 'mean',
                 use_in_batch: bool = False):
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
        n = self.examples_per_group
        if n is None:
            n = int(math.sqrt(logits.size(0)))
        loss = 0
        logits = self.activation(logits)
        logits = logits.view(-1, n)
        targets = torch.ones(logits.size(0)).to(logits.device)
        for idx in self.sample_indices:
            logits_positive = logits[:, idx]
            logits_negative = logits[:, (idx+self.dilation)]
            loss += self.loss_fct(logits_positive, logits_negative)
        return loss / len(self.sample_indices)
