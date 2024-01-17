import logging
from .losses import MSELoss # PointwiseMSE and DistillationMSE
from .losses import PairwiseHingeLoss, GroupwiseHingeLoss
from .losses import CELoss, GroupwiseCELoss # PairwiseCE and GroupwiseCE
from .losses import GroupwiseHingeLossV1, GroupwiseCELossV1
from dataclasses import dataclass

@dataclass
class LossHandler:
    examples_per_group: int = 1
    margin: float = 1
    reduction: str = 'mean'
    stride: int = 1
    dilation: int = 1
    logger: logging = None

    def loss(self, loss_name, batchwise=False):

        loss_fct = None
        n = self.examples_per_group
        # Hinge
        if 'pairwise_hinge' in loss_name:
            self.logger.info("Using objective: PairwiseHingeLoss")
            loss_fct = PairwiseHingeLoss(
                    examples_per_group=n,
                    margin=self.margin, 
                    reduction=self.reduction
            )
        if 'groupwise_hinge' in loss_name:
            self.logger.info("Using objective: GroupwiseHingeLoss")
            loss_fct = GroupwiseHingeLoss(
                    examples_per_group=n, 
                    margin=self.margin,
                    reduction=self.reduction
            )
        if 'groupwise_hinge_v1' in loss_name:
            self.logger.info("Using objective: GroupwiseHingeLossV1")
            loss_fct = GroupwiseHingeLossV1(
                    examples_per_group=n, 
                    margin=self.margin,
                    stride=1, 
                    dilation=1,
                    reduction=self.reduction
            )

        # CE
        if 'pairwise_ce' in loss_name:
            self.logger.info("Using objective: CELoss with Paired")
            loss_fct = CELoss(
                    examples_per_group=self.examples_per_group, 
                    reduction=self.reduction,
                )
        if 'groupwise_ce' in loss_name:
            self.logger.info("Using objective: GroupwiseCELoss")
            loss_fct = GroupwiseCELoss(
                    examples_per_group=n, 
                    reduction=self.reduction
            )
        if 'groupwise_ce_all' in loss_name:
            self.logger.info("Using objective: CELoss")
            loss_fct = CELoss(
                    examples_per_group=n, 
                    reduction=self.reduction,
                    batchwise=batchwise
            )
        if 'groupwise_ce_v1' in loss_name:
            self.logger.info("Using objective: GroupwiseCELossV1")
            loss_fct = GroupwiseCELossV1(
                    examples_per_group=n, 
                    margin=self.margin,
                    stride=self.stride,
                    dilation=self.dilation,
                    reduction=self.reduction
            )

        # BCE MSE (binary) Distillaion-MSE
        if 'pointwise_bce' in loss_name:
            self.logger.info("Using objective: BCELogitsLoss")
            loss_fct = None # default in sentence bert
        if 'pointwise_mse' in loss_name:
            self.logger.info("Using objective: PointwiseMSELoss")
            loss_fct = MSELoss(reduction=self.reduction)
        if 'distillation_mse' in loss_name:
            self.logger.info("Using objective: DistillationMSELoss")
            loss_fct = MSELoss(reduction=self.reduction)

        return loss_fct
