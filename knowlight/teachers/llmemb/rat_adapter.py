import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from sklearn.decomposition import PCA


class RATAdapter(nn.Module):
    """
    Recommendation Adaptation Training (RAT) Two-Layer Trainable Adapter for LLMEmb.
    Implements Equation (6) from LLMEmb paper strictly:
    e_i = W_1 * (W_2 * \bar{e}_i^LLM + b_2) + b_1
    where W_2 in R^{hidden_dim x in_dim}, W_1 in R^{out_dim x hidden_dim}.
    This module's parameters (W1, W2, b1, b2) are trained end-to-end by Adam optimizer 
    during student distillation.
    """

    def __init__(self, in_dim: int = 1536, hidden_dim: int = 512, out_dim: int = 64):
        super(RATAdapter, self).__init__()
        self.fc2 = nn.Linear(in_dim, hidden_dim)   # W2 and b2
        self.fc1 = nn.Linear(hidden_dim, out_dim)  # W1 and b1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # e_i = W_1 * (W_2 * \bar{e}_i^LLM + b_2) + b_1
        return self.fc1(self.fc2(x))


def apply_pca_reduction(
    raw_llm_embeddings: np.ndarray,
    pca_dim: int = 1536
) -> torch.Tensor:
    """
    Level 1 Transformation: Uses PCA to reduce high-dimensional LLM embeddings 
    (e.g. 3584/4096) to an intermediate frozen dimension d_m (e.g. 1536).
    """
    num_items, in_dim = raw_llm_embeddings.shape
    if in_dim > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        reduced_features = pca.fit_transform(raw_llm_embeddings)
    else:
        reduced_features = raw_llm_embeddings

    return torch.tensor(reduced_features, dtype=torch.float32)


def apply_rat_transformation(
    raw_llm_embeddings: np.ndarray,
    pca_dim: Optional[int] = 1536,
    out_dim: int = 64,
    adapter: Optional[RATAdapter] = None
) -> torch.Tensor:
    """
    Applies Two-Level Recommendation Adaptation Training (RAT) transformation:
    1. Level 1: PCA reduction to intermediate size d_m (1536)
    2. Level 2: Two-Layer Adapter MLP projection to target CF dimension (64)
    """
    intermediate_features = apply_pca_reduction(raw_llm_embeddings, pca_dim=pca_dim or 1536)

    if adapter is None:
        hidden_dim = min(intermediate_features.shape[1], 512)
        adapter = RATAdapter(in_dim=intermediate_features.shape[1], hidden_dim=hidden_dim, out_dim=out_dim)

    with torch.no_grad():
        adapted_embeddings = adapter(intermediate_features)

    return adapted_embeddings
