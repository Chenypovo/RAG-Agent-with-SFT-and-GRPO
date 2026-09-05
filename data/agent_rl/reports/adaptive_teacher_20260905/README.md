# Frozen 7B adaptive-teacher experiment: public evidence

This is the compact evidence package for the completed 5 September 2026 run. Scientific outputs were copied byte for byte from the frozen AutoDL results; publication did not rerun generation, training, evaluation or statistical analysis.

**Result:** adaptive-teacher SFT improved the matched answer/evidence metrics. The subsequent original 20-update GRPO reduced retrieval calls but lowered accuracy and retained a strong two-call stopping tendency. E0 historical results retain their original provenance.

- [Full Chinese experiment report](../../../../docs/ADAPTIVE_TEACHER_20260905_REPORT.md)
- [Pilot report](../../../../docs/ADAPTIVE_TEACHER_20260905_PILOT.md)
- [Chronological execution audit](../../../../docs/ADAPTIVE_TEACHER_20260905_AUDIT.md)
- [Exact run configurations](../../../../configs/agent_rl/adaptive_teacher_20260905/)

## Reports and supporting evidence

| Files | Evidence |
|---|---|
| [evaluation/prompt.json](evaluation/prompt.json), [e1_sft.json](evaluation/e1_sft.json), [e2_grpo.json](evaluation/e2_grpo.json) | Complete matched 1,000-task metrics, runtime, peak memory, settings and hashes |
| [paired_comparison.json](paired_comparison.json) | Original paired bootstrap, 2,000 samples, confidence intervals and improved/regressed/tied task counts |
| [paired_behaviour_comparison.json](paired_behaviour_comparison.json) | Actual retrieval attempts, stopping, duplicates and JSON errors, with ratio-of-sums denominators |
| [final_integrity_audit.json](final_integrity_audit.json) | All 3,000 episode checks; same task order, prompt, retrieval and model versions; complete teacher selection reconstruction |
| [stopping_decision_audit.json](stopping_decision_audit.json) | Post-episode evidence labels, conditional next actions and examples selected by fixed task-ID order |
| [pilot50/](pilot50/) and [pilot_cost_gate.json](pilot_cost_gate.json) | 50-task/2-candidate pilot and the original cost gate |
| [e1_teacher/](e1_teacher/) | Full 1,617-task teacher reports, all 4,669 candidate/selected metrics and corrected behaviour audit |
| [e1_sft_data/](e1_sft_data/), [e1_sft/](e1_sft/), [e2_grpo/](e2_grpo/) | Filtering, SFT/GRPO training reports and complete GRPO checkpoint verification |
| [models.json](models.json), [source_hashes.json](source_hashes.json), [stages.json](stages.json) | Model revisions, launch source hashes, exact executed commands and stage durations |
| [real_model_smoke.json](real_model_smoke.json), [reward_audit.json](reward_audit.json), [validation/](validation/) | Actual Blackwell CUDA/7B/QLoRA validation, unchanged reward audit and AutoDL test logs |
| [figures/](figures/) | Training and matched-evaluation stopping distributions as PNG, SVG and CSV |

## Checksums and availability

[PUBLIC_SHA256SUMS](PUBLIC_SHA256SUMS) covers this compact package. Verify it from this directory. The metadata and numerical reports retain their original file contents and internal AutoDL paths.

[RESULTS_SHA256SUMS](RESULTS_SHA256SUMS) is the original manifest for the full 9,874-file, 6.28 GB result bundle. Its own SHA-256 is `d8e82282c8a31aac642d296739fba4902a9de8255edc52e275c987155eed992e`. This manifest is supplied as a content inventory; the large raw trajectories, candidate/RNG journals, adapters and full recovery checkpoints are **not hosted in this Git repository**. They remain in the original local/AutoDL delivery.

[AUTODL_INPUTS_SHA256SUMS](delivery_provenance/AUTODL_INPUTS_SHA256SUMS) identifies 208 base-model, data/index and original E0 input files (23.60 GB). Its paths are relative to the recorded AutoDL run root. Base model weights and indexes are also outside Git. Neither checksum file is a download link, and the existing historical release is not a release of these new adapters.

The complete original artifacts are required to re-run the saved-output audits; this compact package supports inspection of metrics and provenance, not a claim that all raw outputs are downloadable after cloning. Model or scientific reruns are separate experiments and must not be confused with publication verification.

All science, tests, metric calculations and figure generation were performed on AutoDL. Publication used file copies, file checksums and documentation updates. The full frozen experiment report remains unchanged; README updates distinguish current E1/E2 findings from historical E0.

These are one-seed, reused benchmark/dev results. Joint Success means an exact answer plus complete gold sentence evidence. Conditional stopping groups differ across policies; zero empty/failed-retrieval exposure and unlabelled contradictions do not establish recovery. GRPO did not preserve accuracy relative to the new SFT adapter.
