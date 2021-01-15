import torch
from torch import Tensor


def dice_loss(
    pred_score_map: Tensor,
    target_score_map: Tensor,
    train_ignore_mask: Tensor,
    train_boundary_mask: Tensor
) -> Tensor:
    train_mask = train_ignore_mask * train_boundary_mask
    intersection = torch.sum(pred_score_map * target_score_map * train_mask)
    pred, target = pred_score_map * train_mask, target_score_map * train_mask
    union = torch.sum(pred) + torch.sum(target)
    dice_loss = 1.0 - 2 * intersection / union
    return dice_loss
