import pytest

from app.agent_rl.training_checkpoint import save_checkpoint, load_checkpoint


def test_full_checkpoint_resumes_optimizer_update_and_preserves_reference(tmp_path):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = torch.nn.ModuleDict({
                "policy": torch.nn.Linear(2, 1, bias=False),
                "reference": torch.nn.Linear(2, 1, bias=False),
            })
            self.adapters.reference.requires_grad_(False)

    torch.manual_seed(42)
    model = Model()
    optimizer = torch.optim.AdamW(model.adapters.policy.parameters(), lr=0.01)
    x = torch.tensor([[1.0, 2.0]])

    def step(m, opt):
        opt.zero_grad()
        m.adapters.policy(x).square().sum().backward()
        opt.step()

    step(model, optimizer)
    save_checkpoint(tmp_path / "checkpoint", model, optimizer, {"protocol_sha256": "fixed", "completed_updates": 1})
    step(model, optimizer)
    expected = model.adapters.policy.weight.detach().clone()
    resumed = Model()
    reference = resumed.adapters.reference.weight.detach().clone()
    resumed_optimizer = torch.optim.AdamW(resumed.adapters.policy.parameters(), lr=0.01)
    metadata = load_checkpoint(tmp_path / "checkpoint", resumed, resumed_optimizer, protocol_sha256="fixed")
    assert metadata["completed_updates"] == 1
    step(resumed, resumed_optimizer)
    assert torch.equal(expected, resumed.adapters.policy.weight)
    assert torch.equal(reference, resumed.adapters.reference.weight)
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_checkpoint(tmp_path / "checkpoint", resumed, resumed_optimizer, protocol_sha256="changed")
