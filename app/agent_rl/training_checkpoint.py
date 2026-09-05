"""Complete single-GPU GRPO checkpoint, including optimiser and RNG state."""
from __future__ import annotations

import os
from pathlib import Path
import json

from app.agent_rl.artifacts import sha256_file
from app.agent_rl.run_journal import atomic_json, capture_rng, restore_rng


def save_checkpoint(directory, model, optimizer, metadata):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    policy = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if ".policy." in name}
    if not policy:
        raise ValueError("checkpoint has no policy adapter tensors")
    payload = {"policy": policy, "optimizer": optimizer.state_dict(), "rng": capture_rng(), "metadata": metadata}
    temporary = directory / "state.pt.tmp"
    with temporary.open("wb") as handle:
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, directory / "state.pt")
    atomic_json(directory / "manifest.json", {"sha256": sha256_file(directory / "state.pt"), "metadata": metadata})


def load_checkpoint(directory, model, optimizer, *, protocol_sha256):
    import torch
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["metadata"]["protocol_sha256"] != protocol_sha256:
        raise ValueError("GRPO checkpoint protocol mismatch")
    if sha256_file(directory / "state.pt") != manifest["sha256"]:
        raise ValueError("GRPO checkpoint hash mismatch")
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    if state["metadata"] != manifest["metadata"]:
        raise ValueError("GRPO checkpoint metadata mismatch")
    current = model.state_dict()
    if set(state["policy"]) != {name for name in current if ".policy." in name}:
        raise ValueError("GRPO policy adapter tensor names differ")
    with torch.no_grad():
        for name, tensor in state["policy"].items():
            current[name].copy_(tensor)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["metadata"]
