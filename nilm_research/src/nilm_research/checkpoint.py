from __future__ import annotations

from pathlib import Path
import torch

from .model import build_model


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                    epoch: int, best_validation_loss: float, config: dict, transforms: dict,
                    model_kind: str, history: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch, "best_validation_loss": best_validation_loss, "config": config,
        "transforms": transforms, "model_kind": model_kind, "history": history,
    }, target)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["model_kind"], payload["transforms"], payload["config"]["model"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload

