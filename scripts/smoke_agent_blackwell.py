"""Real-model GPU compatibility checks; no scientific dataset or saved adapter."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent_rl.artifacts import dependency_versions, sha256_file
from app.agent_rl.hf_backend import TransformersChatBackend
from app.agent_rl.run_journal import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    import bitsandbytes as bnb
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    started = time.perf_counter()
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    assert "sm_120" in torch.cuda.get_arch_list()
    assert tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 7)
    assert tuple(int(v) for v in torch.version.cuda.split(".")[:2]) >= (12, 8)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    root = Path(args.models_root)
    report = {"status": "running", "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "accelerate_device": str(Accelerator().device), "arch_list": torch.cuda.get_arch_list(),
        "dependencies": dependency_versions(("torch", "transformers", "peft", "accelerate", "bitsandbytes"))}
    atomic_json(output.with_suffix(".progress.json"), report)

    # Load the real 7B once, then keep it resident during the QLoRA check.
    seven = TransformersChatBackend(str(root / "Qwen2.5-7B-Instruct"), dtype="bfloat16", seed=42,
        revision="a09a35458c702b33eeacc393d103063234e8bc28")
    seven.model.requires_grad_(False)
    teacher_complete = seven.make_complete_fn(max_new_tokens=32, temperature=0)
    finaliser_complete = seven.make_complete_fn(max_new_tokens=32, temperature=0)
    report["teacher_output"] = teacher_complete("Return exactly one JSON object.", 'Return {"final_answer":true}.')
    report["finaliser_output"] = finaliser_complete("Answer using the supplied evidence.", "Evidence: Paris is the capital of France. Question: What is the capital of France?")
    assert report["teacher_output"] and report["finaliser_output"]
    report["shared_7b_instance"] = True
    report["7b_trainable_parameters"] = sum(p.numel() for p in seven.model.parameters() if p.requires_grad)

    path = str(root / "Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(path, device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    base.config.use_cache = False
    model = get_peft_model(prepare_model_for_kbit_training(base, use_gradient_checkpointing=True),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": "Return the JSON stop action."},
        {"role": "assistant", "content": '{"final_answer":true}'},
    ], tokenize=False, enable_thinking=False)
    batch = tokenizer(text, return_tensors="pt").to("cuda")
    labels = batch["input_ids"].clone(); labels[:, :-8] = -100
    model.train()
    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)
    optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()
    report.update({"status": "passed", "qlora_loss": float(loss.detach()),
        "qlora_4bit_linear_count": sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()),
        "qlora_finite_nonzero_gradients": True, "qlora_optimizer_step": True,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": "compatibility only; toy prompt; adapter not saved or reused for E1/E2"})
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
