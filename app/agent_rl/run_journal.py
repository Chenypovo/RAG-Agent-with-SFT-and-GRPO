"""Atomic candidate checkpoints with RNG state, without executable serialisation."""
from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from app.agent_rl.policies import PolicyDecision
from app.agent_rl.rollouts import PolicyRollout
from app.agent_rl.verifiers import VerificationResult


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def capture_rng() -> dict:
    import numpy as np
    import torch

    def encoded(tensor):
        return base64.b64encode(tensor.cpu().numpy().tobytes()).decode("ascii")

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu": encoded(torch.get_rng_state()),
        "torch_cuda": [encoded(state) for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    import numpy as np
    import torch

    def decoded(value):
        return torch.from_numpy(np.frombuffer(base64.b64decode(value), dtype=np.uint8).copy())

    random.setstate((state["python"][0], tuple(state["python"][1]), state["python"][2]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(decoded(state["torch_cpu"]))
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("checkpoint GPU count differs from current run")
        torch.cuda.set_rng_state_all([decoded(value) for value in state["torch_cuda"]])


def rollout_from_checkpoint(value: dict) -> PolicyRollout:
    value = dict(value)
    value["decisions"] = tuple(PolicyDecision(**row) for row in value["decisions"])
    value["verification"] = VerificationResult(**value["verification"])
    return PolicyRollout(**value)


class CandidateJournal:
    """One immutable file per completed candidate; restore the last saved RNG."""

    def __init__(self, directory: Path, protocol: dict, *, resume: bool = False):
        self.directory = directory
        self.records = []
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        manifest = directory / "protocol.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError(f"candidate journal already exists: {directory}")
            saved = json.loads(manifest.read_text())
            if saved["sha256"] != fingerprint:
                raise ValueError("resume protocol/input/code mismatch; refusing to rerun or mix candidates")
        elif resume:
            raise FileNotFoundError(f"resume journal is missing: {manifest}")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(manifest, {"sha256": fingerprint, "protocol": protocol})
        for path in sorted(directory.glob("candidate-*.json")):
            envelope = json.loads(path.read_text())
            payload = envelope["payload"]
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if digest != envelope["sha256"]:
                raise ValueError(f"candidate checkpoint hash mismatch: {path}")
            if payload["sequence"] != len(self.records):
                raise ValueError("non-contiguous candidate journal")
            self.records.append(payload)

    def restore(self) -> None:
        if self.records:
            restore_rng(self.records[-1]["rng_after"])

    def candidates_for(self, task_index: int, task_id: str):
        records = [r for r in self.records if r["task_index"] == task_index]
        if any(r["rollout"]["task_id"] != task_id for r in records):
            raise ValueError("task ordering differs from saved journal")
        return [rollout_from_checkpoint(r["rollout"]) for r in records]

    def append(self, task_index: int, candidate_index: int, rollout: PolicyRollout, seconds: float):
        import torch
        record = {
            "sequence": len(self.records), "task_index": task_index,
            "candidate_index": candidate_index, "rollout": asdict(rollout),
            "runtime_seconds": seconds, "rng_after": capture_rng(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        }
        if self.records and task_index < self.records[-1]["task_index"]:
            raise ValueError("cannot append a candidate to an earlier completed task")
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        atomic_json(self.directory / f"candidate-{len(self.records):06d}.json", {"sha256": digest, "payload": record})
        self.records.append(record)
