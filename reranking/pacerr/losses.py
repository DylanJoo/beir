import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from torch.nn import functional as F
from torch.nn import MarginRankingLoss


class PairwiseHingeLoss(nn.Module):
    """
    Compute the loss between two query-document pairs with margin.
    """
    def __init__(self, example_per_group: int = 1, batch_size: int = 1, 
            margin: float = 0, reduction: str = 'mean'):
        super().__init__()
        self.group_size = batch_size // example_per_group
        self.loss_fct = MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, logits: Tensor, labels: Tensor):
        # reshape the logits (B 1)
        logits = logits.view(self.group_size, 2)
        logits_negaitve = logits[:, 0] # see `filter`
        logits_positive = logits[:, 1] # see `filter`
        targets = torch.ones(logits.size(0)).to(logits.device)

        return self.loss_fct(logits_positive, logits_negaitve, targets)

class MarginMSELoss(nn.Module):
    """
    Computes the Margin MSE loss between the query, positive passage and negative passage. This loss
    is used to train dense-models using cross-architecture knowledge distillation setup. 

    Margin MSE Loss is defined as from (Eq.11) in Sebastian Hofstätter et al. in https://arxiv.org/abs/2010.02666:
    Loss(𝑄, 𝑃+, 𝑃−) = MSE(𝑀𝑠(𝑄, 𝑃+) − 𝑀𝑠(𝑄, 𝑃−), 𝑀𝑡(𝑄, 𝑃+) − 𝑀𝑡(𝑄, 𝑃−))
    where 𝑄: Query, 𝑃+: Relevant passage, 𝑃−: Non-relevant passage, 𝑀𝑠: Student model, 𝑀𝑡: Teacher model

    Remember: Pass the difference in scores of the passages as labels.
    """
    def __init__(self, model, scale: float = 1.0, similarity_fct = 'dot'):
        super(MarginMSELoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1) * self.scale
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg

        return self.loss_fct(margin_pred, labels)
