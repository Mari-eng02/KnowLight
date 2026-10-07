import torch
import torch.nn as nn
from knowlight.distillation.loss import ContrastiveEmbeddingAlignmentLoss
from knowlight.teachers.llmemb.rat_adapter import RATAdapter


class ContrastiveDistiller(nn.Module):
    """
    Pipeline 2: Contrastive Embedding Alignment Distiller.
    Implements Joint Loss Function:
    L = L_BPR + lambda_item * L_item + lambda_user * L_user
    where:
    - L_BPR is the standard ranking loss (BPR)
    - L_item is the InfoNCE contrastive alignment loss between LLMEmb and GCN item embeddings (via RATAdapter)
    - L_user is the InfoNCE contrastive alignment loss between LLMEmb user barycenter and GCN user embeddings (via RATAdapter)
    """

    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 512,
        out_dim: int = 64,
        lambda_item: float = 0.1,
        lambda_user: float = 0.1,
        tau: float = 0.2,
        **kwargs
    ):
        super(ContrastiveDistiller, self).__init__()

        self.lambda_item = lambda_item
        self.lambda_user = lambda_user
        if in_dim > out_dim:
            self.adapter = RATAdapter(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        else:
            self.adapter = None
        self.contrastive_loss_fn = ContrastiveEmbeddingAlignmentLoss(tau=tau)

    def forward(
        self,
        bpr_loss: torch.Tensor,
        gcn_item_embeds: torch.Tensor = None,
        llmemb_item_embeds: torch.Tensor = None,
        gcn_user_embeds: torch.Tensor = None,
        llmemb_user_embeds: torch.Tensor = None
    ) -> dict:
        """
        Computes unified contrastive embedding alignment distillation loss and total loss.
        """
        loss_dict = {"l_bpr": bpr_loss.item()}

        l_item = torch.tensor(0.0, device=bpr_loss.device)
        l_user = torch.tensor(0.0, device=bpr_loss.device)

        if gcn_item_embeds is not None and llmemb_item_embeds is not None:
            if self.adapter is not None and llmemb_item_embeds.shape[-1] != gcn_item_embeds.shape[-1]:
                adapted_item = self.adapter(llmemb_item_embeds)
            else:
                adapted_item = llmemb_item_embeds
            l_item = self.contrastive_loss_fn(gcn_item_embeds, adapted_item)

        if gcn_user_embeds is not None and llmemb_user_embeds is not None:
            if self.adapter is not None and llmemb_user_embeds.shape[-1] != gcn_user_embeds.shape[-1]:
                adapted_user = self.adapter(llmemb_user_embeds)
            else:
                adapted_user = llmemb_user_embeds
            l_user = self.contrastive_loss_fn(gcn_user_embeds, adapted_user)

        total_loss = bpr_loss + self.lambda_item * l_item + self.lambda_user * l_user

        loss_dict["l_item"] = l_item.item()
        loss_dict["l_user"] = l_user.item()
        loss_dict["total_loss"] = total_loss

        return loss_dict
