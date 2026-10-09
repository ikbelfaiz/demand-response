import numpy as np
import torch

from nilm_research.data import WindowDataset
from nilm_research.losses import multitask_loss
from nilm_research.model import CausalMultiTaskTCN


def test_output_shapes_nonnegative_and_finite_gradient(prepared, cfg):
    *_, transforms, _, part = prepared
    dataset = WindowDataset([part], 256)
    batch_items = [dataset[i] for i in range(2)]
    batch = {k: torch.stack([x[k] for x in batch_items]) for k in batch_items[0]}
    model = CausalMultiTaskTCN(transforms["appliance_power_scales_w"])
    out = model(batch["inputs"])
    assert out["power_w"].shape == (2, 3)
    assert out["activity_logits"].shape == (2, 3)
    assert torch.all(out["power_w"] >= 0)
    loss, _ = multitask_loss(out, batch, transforms, cfg["loss"])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_causality_future_change_does_not_change_earlier_encoding(prepared):
    *_, transforms, _, part = prepared
    model = CausalMultiTaskTCN(transforms["appliance_power_scales_w"]).eval()
    x = torch.from_numpy(part.inputs[:, :512]).unsqueeze(0)
    changed = x.clone(); changed[:, :, 400:] += 100.0
    with torch.no_grad():
        first = model.encode_sequence(x)
        second = model.encode_sequence(changed)
    torch.testing.assert_close(first[:, :, :400], second[:, :, :400], rtol=0, atol=0)
    assert not torch.allclose(first[:, :, 450:], second[:, :, 450:])


def test_receptive_field():
    assert CausalMultiTaskTCN.receptive_field == 253

