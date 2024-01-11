<h2>The settings of PACE-RR</h2>

## Filters

1. Identity
2. Boundary
    - params: num: int = 1
    - params: random-sample: int = 0
3. 


## Objectives

Primary losses
1. BCELogitLoss

2. PairwiseHingeLoss:
Compute the loss between two query-document pairs with margin.

3. LCELoss (Localized Contrastive Estimation)

Composite loss
1. CompositeV1Loss: BCELogitLoss + PairwiseHingeLoss
2. CompositeV2Loss: BCELogitLoss + LCELoss
2. CompositeV3Loss: BCELogitLoss + PairwiseHingeLoss + LCELoss