# PhD CV Entry: Agentic RAG / Agent RL

**Reproducible Post-Training and Evaluation of a Multi-Turn Retrieval Agent**

- Unified the Agentic RAG and Agent RL retrieval paths around shared LanceDB, BM25, reciprocal-rank fusion, and BGE cross-encoder components; selected a `15 → 6` candidate/reranking configuration through a 1,000-item HotpotQA ablation, improving complete sentence-evidence retrieval from 34.7% (BM25@8) to 59.5% (hybrid-rerank@6).

- Post-trained Qwen3-1.7B controllers via QLoRA SFT and 20-update multi-turn GRPO with same-state grouped rollouts, action-token-only clipped objectives, frozen-reference KL regularization, reward-hacking audits, numerical checks, and end-to-end artifact provenance.

- Upgraded the controlled evaluation stack with hybrid retrieval, BGE reranking, and a frozen Qwen2.5-7B finalizer; on 1,000 paired tasks, the GRPO system improved Answer EM from 19.9% to 42.5% (95% CI: +19.6 to +25.5 pp), Joint Success from 11.9% to 35.0%, and complete sentence evidence from 37.4% to 65.5%, reported explicitly as a combined system-stack effect rather than an RL-only gain.

- Within the upgraded stack, GRPO reduced mean tool calls from 5.000 to 2.121 (-57.6%), duplicate-call rate from 61.78% to 1.28%, and budget exhaustion from 100% to 0 while preserving Answer EM and Joint Success (paired 95% CIs spanning zero); scoped conclusions to a single-seed, reused development set.
