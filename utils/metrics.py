import torch
from typing import List, Dict

def get_correct_top_k_counts(correctly_predicted_items_with_rank: torch.LongTensor, num_items_to_predict: List[int]) -> torch.Tensor:
    X = correctly_predicted_items_with_rank
    device = X.device
    B, I = X.size()
    max_items_to_predict = max(num_items_to_predict)
    rank_correct_mask = torch.zeros((B, I + 1), dtype=torch.int64, device=device).scatter_add(
        1, X + 1, torch.ones((B, I + 1), dtype=torch.int64, device=device))[:, 1: max_items_to_predict + 1]
    return rank_correct_mask.cumsum(dim=1), rank_correct_mask

def calculate_top_N_metrics(predicted_ranks: torch.LongTensor, items_yet_to_be_interacted_with: torch.BoolTensor, items_encountered_previously: torch.BoolTensor,
                            num_items_to_predict: List[int]) -> Dict[str, float]:
    device = predicted_ranks.device
    correctly_predicted_items = (predicted_ranks + 1) * items_yet_to_be_interacted_with - 1
    correctly_predicted_new_items = (correctly_predicted_items + 1) * (1 - items_encountered_previously) - 1

    correctly_predicted_counts, correctly_predicted_ranks = get_correct_top_k_counts(correctly_predicted_items, num_items_to_predict)
    correctly_new_predicted_counts, correctly_new_predicted_ranks = get_correct_top_k_counts(correctly_predicted_new_items, num_items_to_predict)

    num_items_to_be_encountered = items_yet_to_be_interacted_with.sum(dim=1)
    num_new_items_to_be_encountered = ((1 - items_encountered_previously) * items_yet_to_be_interacted_with).sum(dim=1)

    recall = correctly_predicted_counts.float() / num_items_to_be_encountered.unsqueeze(1).float()
    recall_new = torch.nan_to_num(correctly_new_predicted_counts.float() / num_new_items_to_be_encountered.unsqueeze(1).float(), nan=1.0)
    max_items_to_predict = max(num_items_to_predict)
    precision = correctly_predicted_counts.float() / torch.arange(1, max_items_to_predict + 1, device=device).unsqueeze(0).float()
    precision_new = correctly_new_predicted_counts.float() / torch.arange(1, max_items_to_predict + 1, device=device).unsqueeze(0).float()

    map = (precision * correctly_predicted_ranks.float()).cumsum(dim=1) / torch.arange(1, max_items_to_predict + 1, device=device).unsqueeze(0).float()
    map_new = (precision_new * correctly_new_predicted_ranks.float()).cumsum(dim=1) / torch.arange(1, max_items_to_predict + 1, device=device).unsqueeze(0).float()

    recall = recall.sum(dim = 0)
    recall_new = recall_new.sum(dim = 0)
    map = map.sum(dim = 0)
    map_new = map_new.sum(dim = 0)
    precision = precision.sum(dim = 0)
    precision_new = precision_new.sum(dim = 0)

    metrics = {}
    for N in num_items_to_predict:
        metrics[f'recall@{N}'] = recall[N - 1]
        metrics[f'recall_new@{N}'] = recall_new[N - 1]
        metrics[f'map@{N}'] = map[N - 1]
        metrics[f'map_new@{N}'] = map_new[N - 1]
        metrics[f'precision@{N}'] = precision[N - 1]
        metrics[f'precision_new@{N}'] = precision_new[N - 1]

    return metrics