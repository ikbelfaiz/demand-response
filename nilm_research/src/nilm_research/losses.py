from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def multitask_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                   transforms: dict, loss_cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["target_valid"].bool()
    scales = outputs["power_w"].new_tensor(transforms["appliance_power_scales_w"])
    target_scaled = batch["power_w"] / scales
    power_element = F.huber_loss(outputs["power_scaled"], target_scaled, reduction="none",
                                 delta=float(loss_cfg["huber_delta_scaled"]))
    power_losses = torch.stack([_masked_mean(power_element[:, j], mask[:, j]) for j in range(3)])
    power_loss = power_losses.mean()

    pos_weight = outputs["power_w"].new_tensor(transforms["activity_positive_class_weights"])
    activity_element = F.binary_cross_entropy_with_logits(
        outputs["activity_logits"], batch["activity"], reduction="none", pos_weight=pos_weight
    )
    activity_losses = torch.stack([_masked_mean(activity_element[:, j], mask[:, j]) for j in range(3)])
    activity_loss = activity_losses.mean()

    excess_w = torch.relu(outputs["power_w"].sum(dim=1) - batch["aggregate_w"]
                          - float(transforms["consistency_tolerance_w"]))
    aggregate_scale = max(float(transforms["aggregate_std_w"]), 1.0)
    consistency_loss = F.smooth_l1_loss(excess_w / aggregate_scale, torch.zeros_like(excess_w))
    total = (float(loss_cfg["power_weight"]) * power_loss
             + float(loss_cfg["activity_weight"]) * activity_loss
             + float(loss_cfg["consistency_weight"]) * consistency_loss)
    components = {
        "loss": float(total.detach()), "power_loss": float(power_loss.detach()),
        "activity_loss": float(activity_loss.detach()), "consistency_loss": float(consistency_loss.detach()),
    }
    for j, name in enumerate(("ac", "water_heater", "washing_machine")):
        components[f"power_{name}"] = float(power_losses[j].detach())
        components[f"activity_{name}"] = float(activity_losses[j].detach())
    return total, components

