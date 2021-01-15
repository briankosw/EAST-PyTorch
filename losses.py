import torch
from torch import Tensor


def dice_loss(
    pred_score_map: Tensor,
    target_score_map: Tensor,
    train_ignore_mask: Tensor,
    train_boundary_mask: Tensor,
) -> Tensor:
    train_mask = train_ignore_mask * train_boundary_mask
    intersection = torch.sum(pred_score_map * target_score_map * train_mask)
    pred, target = pred_score_map * train_mask, target_score_map * train_mask
    union = torch.sum(pred) + torch.sum(target)
    dice_loss = 1.0 - 2 * intersection / (union + 1e-6)
    return dice_loss


def rbox_loss(
    pred_geometry_map: Tensor,
    target_geometry_map: Tensor,
    train_ignore_mask: Tensor,
    train_boundary_mask: Tensor,
    angle_lambda: int = 10
) -> Tensor:
    pred_top, pred_right, pred_bottom, pred_left, pred_angle = torch.split(
        pred_geometry_map, split_size_or_sections=[1, 1, 1, 1, 1], dim=2
    )
    target_top, target_right, target_bottom, target_left, target_angle = torch.split(
        target_geometry_map, split_size_or_sections=[1, 1, 1, 1, 1], dim=2
    )
    pred_area = (pred_top + pred_bottom) * (pred_left + pred_right)
    target_area = (target_top + target_bottom) * (target_left + target_right)
    h_inter = torch.minimum(
        pred_right, target_right) + torch.minimum(pred_left, target_left
    )
    w_inter = torch.minimum(
        pred_top, target_top) + torch.minimum(pred_bottom, target_bottom
    )
    intersection = h_inter * w_inter
    union = pred_area + target_area - intersection
    box_loss = -torch.log((intersection + 1e-6) / (union + 1e-6))
    angle_loss = 1 - torch.cos(pred_angle - target_angle)
    rbox_loss = box_loss + angle_lambda * angle_loss
    train_mask = torch.unsqueeze(train_ignore_mask * train_boundary_mask, dim=2)
    rbox_loss = torch.mean(torch.sum(rbox_loss * target_geometry_map * train_mask))
    return rbox_loss
