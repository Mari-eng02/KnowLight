import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreDistillationLoss(nn.Module):
    """
    Score Knowledge Distillation Loss (Pipeline 1).
    Imitates LLM soft probabilities via Mean Squared Error against GCN dot products.
    """

    def __init__(self):
        super(ScoreDistillationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        llm_scores: torch.Tensor,
        gcn_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            llm_scores: Continuous probability scores extracted from LLM (batch_size,)
            gcn_scores: Sigmoid of LightGCN dot products (e_u * e_i) (batch_size,)
        """
        return self.mse_loss(gcn_scores, llm_scores)


class ContrastiveEmbeddingAlignmentLoss(nn.Module):
    """
    InfoNCE / NT-Xent Contrastive Loss for Embedding Space Alignment (Pipeline 2).
    Aligns student GCN embeddings (user or item) with teacher LLMEmb embeddings.
    """

    def __init__(self, tau: float = 0.2):
        super(ContrastiveEmbeddingAlignmentLoss, self).__init__()
        self.tau = tau

    def forward(
        self,
        gcn_embeds: torch.Tensor,
        llmemb_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            gcn_embeds: Tensor of shape (batch_size, dim) from LightGCN
            llmemb_embeds: Tensor of shape (batch_size, dim) from LLMEmb
        Returns:
            Scalar InfoNCE contrastive alignment loss
        """
        if gcn_embeds is None or llmemb_embeds is None or gcn_embeds.shape[0] == 0:
            return torch.tensor(0.0, device=gcn_embeds.device if gcn_embeds is not None else "cpu")

        # Normalize embeddings for cosine similarity
        gcn_norm = F.normalize(gcn_embeds, p=2, dim=-1)
        llmemb_norm = F.normalize(llmemb_embeds, p=2, dim=-1)

        # Cosine similarity matrix [batch_size, batch_size]
        sim_matrix = torch.matmul(gcn_norm, llmemb_norm.T) / self.tau

        # Positive pairs are on the diagonal
        labels = torch.arange(gcn_embeds.shape[0], device=gcn_embeds.device)

        # Cross-entropy loss over similarity matrix
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# Alias for convenience
InfoNCELoss = ContrastiveEmbeddingAlignmentLoss

