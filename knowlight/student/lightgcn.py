import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class LightGCN(nn.Module):
    """
    LightGCN Student Model for Recommendation Systems.
    Paper: 'LightGCN: Simplifying and Powered Graph Convolution Network for Recommendation' (SIGIR 2020)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        keep_prob: float = 0.6
    ):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.keep_prob = keep_prob

        # Initialize base structural embeddings E_0
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.Graph = None  # Normalized adjacency matrix (PyTorch sparse tensor)

    def set_graph(self, interaction_matrix: sp.csr_matrix, device: torch.device):
        """Construct normalized bipartite adjacency matrix A_hat."""
        adj_mat = sp.dok_matrix(
            (self.num_users + self.num_items, self.num_users + self.num_items),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = interaction_matrix.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.tocsr()

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
        nonzero_mask = rowsum > 0
        d_inv_sqrt[nonzero_mask] = np.power(rowsum[nonzero_mask], -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        norm_adj = norm_adj.tocoo()

        indices = torch.from_numpy(
            np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
        )
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)

        self.Graph = torch.sparse_coo_tensor(indices, values, shape).to(device)

    def computer(self):
        """Perform LightGCN multi-layer graph propagation."""
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.Graph is not None:
                all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        """
        Compute structural embeddings z_u, z_i and ranking scores r = z_u^T z_i
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        
        # Dot product ranking score r_gcn
        scores = torch.sum(users_emb * items_emb, dim=-1)
        return users_emb, items_emb, scores

    def get_user_embedding(self, users: torch.Tensor):
        """Get propagated final layer embeddings for given user indices."""
        all_users, _ = self.computer()
        return all_users[users]

    def get_item_embedding(self, items: torch.Tensor):
        """Get propagated final layer embeddings for given item indices."""
        _, all_items = self.computer()
        return all_items[items]

    @property
    def can_score_items_together(self) -> bool:
        return True

    def get_all_user_embeddings(self) -> torch.Tensor:
        """Get propagated final layer embeddings for all users."""
        all_users, _ = self.computer()
        return all_users

    def get_all_item_embeddings(self) -> torch.Tensor:
        """Get propagated final layer embeddings for all items."""
        _, all_items = self.computer()
        return all_items

    def score_items(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product ranking scores between users and items.
        Supports 1D item tensor (B,) or 2D item tensor (B, K).
        """
        all_users, all_items = self.computer()
        u_emb = all_users[users]
        if items.dim() == 1:
            i_emb = all_items[items]
            return torch.sum(u_emb * i_emb, dim=-1)
        elif items.dim() == 2:
            i_emb = all_items[items]
            return torch.sum(u_emb.unsqueeze(1) * i_emb, dim=-1)
        else:
            i_emb = all_items[items]
            return torch.sum(u_emb.unsqueeze(1) * i_emb, dim=-1)

    def predict_all_items(self, users: torch.Tensor):
        """Compute ranking scores for given users across all items."""
        all_users, all_items = self.computer()
        users_emb = all_users[users] # (batch_size, dim)
        scores = torch.matmul(users_emb, all_items.T) # (batch_size, num_items)
        return scores

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        decay: float = 1e-4
    ):
        """Calculate Bayesian Personalized Ranking (BPR) loss."""
        all_users, all_items = self.computer()
        
        u_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=-1)

        # BPR loss: -log(sigmoid(pos_scores - neg_scores))
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # L2 regularization on initial 0-th layer embeddings
        u_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)

        reg_loss = (1 / 2) * (
            u_emb_0.norm(2).pow(2)
            + pos_emb_0.norm(2).pow(2)
            + neg_emb_0.norm(2).pow(2)
        ) / float(len(users))

        return bpr_loss + decay * reg_loss
