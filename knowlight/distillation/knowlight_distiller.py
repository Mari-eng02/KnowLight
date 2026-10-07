import torch
import torch.nn as nn
import torch.nn.functional as F
from knowlight.distillation.loss import ScoreDistillationLoss


class KnowLightDistiller(nn.Module):
    """
    KnowLight Score Distillation Coordinator.
    Implements Joint Loss Function L = alpha * L_R + beta * L_D where:
    - L_R is the ranking loss (BPR) with alpha fixed (default 1.0)
    - L_D is the Score Distillation loss (MSE between LLM scores and GCN scores)
    - beta is an explored hyperparameter for tuning distillation strength.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        **kwargs
    ):
        super(KnowLightDistiller, self).__init__()

        self.alpha = alpha
        self.beta = beta

        # Score Distillation Loss
        self.semantic_loss_fn = ScoreDistillationLoss()

    def forward(
        self,
        bpr_loss: torch.Tensor,
        llm_scores: torch.Tensor = None,
        gcn_scores: torch.Tensor = None
    ) -> dict:
        """
        Computes the unified score distillation loss and total loss.
        """
        loss_dict = {"l_bpr": bpr_loss.item()}

        if llm_scores is not None and gcn_scores is not None:
            # Score Distillation Loss (L_D)
            l_d = self.semantic_loss_fn(llm_scores, gcn_scores)
        else:
            l_d = torch.tensor(0.0, device=bpr_loss.device)

        # Total Loss: L = alpha * L_BPR + beta * L_D
        total_loss = self.alpha * bpr_loss + self.beta * l_d

        loss_dict["l_d"] = l_d.item()
        loss_dict["total_loss"] = total_loss
        
        return loss_dict

