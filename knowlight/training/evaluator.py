import torch
from knowlight.utils.metrics import compute_topk_metrics


class Evaluator:
    """
    Evaluator for ranking recommendation performance on validation and test sets.
    """

    def __init__(self, top_k_list: list = [5, 10, 20]):
        self.top_k_list = top_k_list

    def evaluate(self, model, data_loader, test_pos_items: dict, device: torch.device, is_test: bool = None) -> dict:
        """
        Evaluate LightGCN model on positive items for all users.
        """
        model.eval()
        all_metrics = {}

        users = list(test_pos_items.keys())
        pos_items_list = [test_pos_items[u] for u in users]

        users_tensor = torch.tensor(users, dtype=torch.long).to(device)

        # Auto-detect test split if not explicitly specified
        if is_test is None and data_loader is not None:
            is_test = (test_pos_items is getattr(data_loader, "test_pos_items", None))

        with torch.no_grad():
            # Get ranking scores for all items
            all_scores = model.predict_all_items(users_tensor) # (num_eval_users, num_items)

            # Mask out already interacted training items (and validation items during test) for each user
            if data_loader is not None and hasattr(data_loader, "train_matrix"):
                for idx, u in enumerate(users):
                    train_items = data_loader.train_matrix[u].indices
                    if len(train_items) > 0:
                        all_scores[idx, train_items] = -1e9

                    # If evaluating on test split, also mask out validation items (seen history)
                    if is_test and hasattr(data_loader, "val_pos_items") and u in data_loader.val_pos_items:
                        val_items = data_loader.val_pos_items[u]
                        if len(val_items) > 0:
                            all_scores[idx, val_items] = -1e9

            metrics = compute_topk_metrics(
                pos_items=pos_items_list,
                rank_scores=all_scores,
                top_k_list=self.top_k_list
            )

        return metrics
