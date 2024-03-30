import torch
from typing import Union, Tuple, List, Iterable, Dict
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

class PointwiseInbatchBCELoss_test(nn.Module):
    def __init__(self, 
                 examples_per_group: int = 1, 
                 reduction: str = 'mean', 
                 batch_size: int = None, 
                 negative_selection: 'str' = 'all'):
        super().__init__()
        self.examples_per_group = examples_per_group
        self.loss_fct = BCEWithLogitsLoss(reduction=reduction)
        self.batch_size = batch_size
        self.negative_selection = negative_selection
        assert (negative_selection != 'all') == (reduction == 'none'), \
                'reduction should be the none thus negative selection.'

    def forward(self, logits: Tensor, labels: Tensor):
        if self.batch_size:
            n_rows = self.batch_size * 1 # this can be more than 1
            logits = logits.view(n_rows, -1) 
        else:
            n_cols = self.examples_per_group
            logits = logits.view(-1, n_cols) # reshape (B n). this is document-centirc
        targets = torch.zeros(logits.size()).to(logits.device)
        targets[:, 0] = 1.0

        # pooled the logits and targets into one list
        loss = self.loss_fct(logits.view(-1), targets.view(-1))
        loss_matrix = loss.view(targets.size())

        if self.negative_selection == 'hard':
            loss = loss_matrix[:, 0].mean()
            loss += loss_matrix[:, 1:].max(-1).values.mean()
            return loss / 2
        elif self.negative_selection == 'mean':
            loss = loss_matrix[:, 0].mean()
            loss += loss_matrix[:, 1:].mean()
            return loss / 2
        elif self.negative_selection == 'random':
            B, N = targets.size(0), targets.size(1)
            samples = 1+torch.randint(N-1, (B, 1), device=logits.device)
            loss = loss_matrix[:, 0].mean()
            loss += loss_matrix.gather(1, samples).mean()
            return loss / 2
        else:
            return loss

# class LocalizedIBCELoss(nn.Module):
#
#     def __init__(self, 
#                  examples_per_group: int = 1, 
#                  margin: float = 1, 
#                  stride: int = 1,    # the size between selected positions
#                  dilation: int = 1,  # the position of the paired negative 
#                  reduction: str = 'mean',
#                  use_in_batch: bool = False):
#         super().__init__()
#         self.examples_per_group = examples_per_group
#         self.loss_fct = CrossEntropyLoss(reduction=reduction)
#         self.activation = nn.Sigmoid()
#         self.stride = stride
#         self.dilation = dilation
#         self.sample_indices = list(
#                 range(0, examples_per_group-dilation, stride)
#         )
#         for i, idx in enumerate(self.sample_indices):
#             print(f"The {i+1} pair: + {idx}; - {idx+dilation}")
#
#     def forward(self, logits, labels):
#         n = self.examples_per_group
#         if n is None:
#             n = int(math.sqrt(logits.size(0)))
#         loss = 0
#         logits = self.activation(logits)
#         logits = logits.view(-1, n)
#         targets = torch.ones(logits.size(0)).to(logits.device)
#         for idx in self.sample_indices:
#             logits_positive = logits[:, idx]
#             logits_negative = logits[:, (idx+self.dilation)]
#             loss += self.loss_fct(logits_positive, logits_negative)
#         return loss / len(self.sample_indices)
