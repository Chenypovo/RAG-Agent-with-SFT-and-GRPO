# Frozen 7B adaptive teacher: 2026-09-05 execution audit

Status: full teacher generation, QLoRA SFT, all20 original GRPO updates, all three1,000-task evaluations and paired statistics completed. Final scientific integrity audits passed. Entries below are a chronological record; earlier pending-stage notes are superseded by their later completion entries.

## Baseline and authorisation

- Local HEAD and the user-supplied GitHub repository's master both resolve to `19458b5fbd3a37d44f8673c1b0f0da7c65c122ea`.
- No filesystem AGENTS.md was found in the repository or its ancestor directories. The AGENTS instructions supplied in the conversation apply.
- All tests and experimental execution take place on AutoDL. Local activity is limited to reading, editing, file integrity checks and transferring artifacts.

## Environment observed on AutoDL

- Host: `<gpu-host>`.
- Single NVIDIA GeForce RTX 5090 D, 32,607 MiB reported by nvidia-smi.
- Driver 595.71.05; CUDA toolkit 12.8.93 at `/usr/local/cuda/bin/nvcc`.
- Python 3.12.3; PyTorch 2.8.0+cu128; compiled architectures include sm_120; device capability (12, 0).
- BF16 CUDA matrix multiplication completed and returned finite values.
- The initial data disk was empty. Transformers, Accelerate, PEFT and bitsandbytes were initially absent. Installed into an isolated environment using the existing CUDA-enabled PyTorch: Transformers 4.57.6, Accelerate 1.14.0, PEFT 0.20.0, bitsandbytes 0.50.0 and LanceDB 0.36.0.
- Accelerate detects CUDA. A bitsandbytes NF4 linear-layer forward/backward GPU check returned finite outputs and gradients. This is a kernel check, not the required actual Qwen3 QLoRA or 7B generation smoke test; those remain pending model availability.
- The supplied GitHub release exists but its assets list is empty. It does not provide downloadable base models or historical adapter attachments. The user subsequently authorised downloading the required models and continuing the experiment.
- Fixed model revisions: Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`; Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; bge-small-en-v1.5 `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; bge-reranker-base `2cfc18c9415c912f9d8155881c133215df768a70`.
- Direct Hugging Face access was initially unreachable. The mirror and official ModelScope were reachable but slow. Corresponding Qwen ModelScope weight hashes matched the fixed Hugging Face files. AutoDL's supplied network helper then restored official Hugging Face access. Transfers resumed existing partial files; these were provisioning changes, not scientific reruns. Logs and final file hashes are retained.

## Gaps and planned changes

1. Teacher generation and GRPO use BM25 directly. Add shared configuration wiring around the existing `build_hybrid_registry` and use LanceDB + BM25 + RRF + BGE, with 15 candidates, 6 returned sentences and RRF k=60. No parent-child expansion.
2. Enforce one teacher/finaliser model path and revision and verify object reuse. Cap teacher candidates at four and actions at five.
3. Add incremental candidate/trajectory storage, recoverable checkpoints, exact random-state capture, timing and peak memory measurements before the pilot.
4. Add selected/all-candidate behaviour audits, retaining raw failures and explicit denominators. Contradictory-retrieval recovery must remain unresolved unless contradiction labels are actually available; do not invent a detector or fabricate exposure counts.
5. GRPO currently saves adapters without complete optimiser/random-state recovery. Add safe checkpoint support before training if required for continuation.
6. The observation renderer and controller prompt have changed since E0. Historical prompt-only evaluation cannot be reused as this run's matched baseline. Run prompt-only, E1 SFT and E2 SFT+GRPO under one frozen new protocol; report E0 separately.

Files: `app/agent_rl/retrieval_setup.py`, `scripts/build_agent_teacher_rollouts.py`, `scripts/train_agent_grpo.py`, new preparation tests; subsequent checkpoint/diagnostic helpers and run-specific configuration/report files. Preserve existing teacher/policy edits.

## Preparation validation

- Retrieval wiring, shared-model enforcement and candidate/action caps are implemented. Legacy scripted generation still defaults to BM25; adaptive generation defaults to hybrid-rerank. GRPO accepts the same retrieval configuration without modifying the historical GRPO JSON.
- Initial remote test attempt: 63 passed, 1 failed because the fixed `smoke_tasks.jsonl` fixture had not been copied. Copied the original fixture unchanged and reran the same tests; no scientific experiment was run or retried.
- Corrected remote test run: 64 passed, 2 deprecation warnings, in 1.97 seconds. This validates code/configuration behaviour only.
- Existing E0 reports pass SHA-256 revalidation. All local initial interview-note files remain untouched.
- Implemented atomic per-candidate JSON checkpoints including Python/NumPy/Torch CPU/CUDA random states and content hashes. Completed candidates are reused on an explicit resume; changed protocols or corrupt checkpoints are rejected. Selected and all-candidate trajectories, candidate decisions and timings are retained.
- Implemented complete per-update GRPO checkpoints including policy adapter, optimiser, random state and verified trajectory prefix. Resume preserves the initial SFT reference adapter and archives any uncommitted tail.
- Added evaluation episode checkpoints, stage runtime/peak CUDA tracking, retrieval-count distributions, JSON invalidity, premature stopping and recovery counts with explicit denominators. Unlabelled contradiction recovery remains unresolved.
- Latest AutoDL preparation suite: 106 passed, 2 deprecation warnings. Tests include optimiser-resume equivalence, candidate resumption without resampling and the full-budget 24-hour cost gate. No teacher pilot, SFT, GRPO or benchmark evaluation has started as of this preparation entry.

## Model readiness update

- All four model snapshots passed file size and content verification against the frozen revisions. LFS files were checked against their SHA-256 values; other files were checked against Git blob hashes and assigned SHA-256 values. The complete manifest is `audit/model_download_result.json` on AutoDL and in the local audit mirror.
- Remaining large files were transferred from the official ModelScope repository only where the content matched the frozen Hugging Face files; exact byte ranges and the final complete hashes were verified. Download transport changes did not change model contents or repeat any scientific experiment.
- Real-model smoke test passed: one frozen Qwen2.5-7B instance generated teacher and finaliser outputs; the same instance remained resident while Qwen3-1.7B NF4 QLoRA completed finite, nonzero-gradient backpropagation and an optimiser step. The smoke adapter was not saved or reused.
- Smoke measurement: 6.58 seconds inside the test, 11.51 seconds including process startup; peak allocated 17,990,410,752 bytes and peak reserved 18,918,408,192 bytes. The 7B model had zero trainable parameters; Qwen3 had 196 4-bit linear layers.
- The staged driver is running remotely, with one process per stage and an automatic stop on an error or a pilot estimate above 24 hours. Per-candidate and per-update checkpoints support an explicitly audited recovery without resampling completed work.

## Pilot completion and continuation

- The pilot generated 91 candidates for 50 tasks in 275.85 seconds including setup. All 451 prompts matched the observation-only renderer, all candidate hashes passed and all 50 selections were independently reproduced.
- Full maximum-candidate teacher time is estimated at 5.26 hours; remaining workflow at 9.71 hours, with an 11.76-hour conservative estimate based on the measured candidate 90th percentile and an explicit training/startup allowance. Full generation started with the original authorised scope.
- Selected pilot trajectories achieved 68% Complete Sentence Evidence and 46% Joint Success, but 58% exhausted the action budget. These are training-pilot observations, not new benchmark/CV results. Details are in `ADAPTIVE_TEACHER_20260905_PILOT.md`.
- A postprocessing classification defect was found: an unsupported tool named `final_answer` was labelled like a genuine stop. The environment had correctly rejected those actions. The diagnostic helper now separates unsupported names and distinguishes actual retrieval attempts from the historical generic MeanToolCalls. Original reports/code were preserved, and 4 focused regression tests passed on AutoDL. No generation, selection, reward or seed was changed; no scientific trajectory was rerun.

## Frozen experiment plan

- Seed 42. Pilot: first 50 training tasks, at most 2 candidates/task, max_steps=5.
- Full teacher: all 1,617 training tasks, at most 4 candidates/task, max_steps=5; endpoint verifier ranking and existing clean-complete-evidence candidate early stop.
- Continue only after the pilot supports an estimate within 24 hours. Do not reduce task count, candidates, steps, verifier strictness or reward to meet the budget.
- Qwen3-1.7B NF4 QLoRA SFT on complete-sentence-evidence selected trajectories, without the legacy pre-stop duplication.
- GRPO: existing seed/reward, 20 updates, 2 tasks/update, 4 rollouts/task, 128-task pool, 2 optimisation epochs, unchanged learning rate/clip/KL coefficients.
- Fixed 1,000-question HotpotQA benchmark/dev; same hybrid retrieval, frozen 7B finaliser and seed for all new controllers. Paired bootstrap with 2,000 resamples, seed 42.
- Only retrieve_docs and final_answer are available to the policies. E0 numerical claims retain their historical provenance.

## Full teacher and SFT completion

- Full teacher: all 1,617 tasks completed, 4,669 candidates, mean 2.887 candidates/task. Candidate-count histogram: 407 tasks used one candidate, 222 used two, 134 used three and 854 used all four. Total stage time including startup was 13,863.26 seconds (3 h 51 min).
- Independent postprocessing audit passed all candidate SHA-256 checks, candidate/seed bounds, exact reconstruction of all 1,617 selections and observation-only rendering of all 23,028 decisions. Saved separately as `results/e1_teacher/integrity_audit.json`; original reports are retained.
- Selected training-trajectory metrics: Answer EM64.25%, F1 71.84%, Joint Success59.31%, Complete Sentence Evidence85.53%, Complete Document Evidence90.72%; mean actual retrieval attempts4.053. These are verifier-selected training data metrics, not new benchmark/CV figures.
- Selected retrieval histogram: 2:57, 3:222, 4:917, 5:421. Decision lengths: 3:43, 4:176, 5:1,398. Selected budget exhaustion31.73%. Thus the fixed two-call target is absent from most new trajectories, but dependence on the maximum action budget remains a material concern.
- After two retrieval attempts with incomplete gold evidence, 417 selected trajectories continued, 13 stopped and7 emitted an invalid next action;201 continuations eventually obtained complete evidence and133 ended with Joint Success. These are post-episode labels.
- Actual SFT dataset:949 trajectories and4,541 examples (3,697 retrieval decisions and844 explicit stops), no augmentation. The original filter retained105 complete-evidence budget-exhausted trajectories without an explicit stop. It discarded399 duplicate-containing trajectories,103 with invalid-action transitions, one parse failure and165 below the evidence threshold. No filtering rule was changed after observing this.
- SFT-kept retrieval histogram: 2:31, 3:142, 4:671, 5:105;81.77% use4+ calls. All kept transitions are parse/duplicate clean, so the SFT subset does not directly teach recovery from these failures. Contradiction recovery still lacks adjudicated labels; empty/failed retrieval exposure was zero in full teacher generation as well.
- QLoRA SFT completed two epochs /568 optimiser steps, training loss0.1955, training time2,461.83 seconds and total stage time2,489.12 seconds. Of4,541 prompts,749 were truncated under the unchanged2,048-token limit. Peak allocated memory7,019,414,016 bytes; peak reserved16,716,398,592 bytes. Final adapter and checkpoints550/568 were saved, following the original save_total_limit=2 configuration.
- GRPO has started from this new SFT adapter; the first four updates and full recovery states were confirmed. No scientific stage has been restarted. Continue to all20 updates and matched evaluations before drawing downstream conclusions.

## GRPO completion and evaluation start

- Completed all20 updates,160 episodes and714 decisions from the unchanged128-task training pool, seed42. Training time1,125.41 seconds; total process time1,142.15 seconds; stage wall time1,145.17 seconds including startup. Peak allocated24,912,108,544 bytes and peak reserved27,621,588,992 bytes. There were113 truncated action prompts under the unchanged2,048-token setting.
- An independent read-only audit verified finite training metrics, an unchanged frozen SFT reference, all20 complete checkpoint hashes and exact equality of all392 final adapter tensors with the update20 checkpoint. The final adapter SHA-256 is `8a9e754adf00cb7cfc5d15c99f06a3bf5fba53fc39e276b60fa93c868058bddb`.
- The initial read-only audit tried the wrong report location for `reference_unchanged` (`provenance` instead of `policy`) and raised KeyError before producing output. The field path was corrected in the saved `audit_grpo_outputs.py`; the complete audit then passed. Recorded under `audit/postprocessing_fixes.jsonl`. No model output or scientific stage was rerun.
- Prompt-only1,000-task evaluation has begun. It will be followed by E1 SFT and E2 GRPO in the same hybrid+7B environment, then the original paired headline comparison and a supplemental paired behaviour comparison that preserves per-decision denominators. Two enumerated-bootstrap tests for the latter passed on AutoDL.
- Runtime values measure the actual operational run, including concurrent read-only audits/backups; they should not be presented as isolated inference-latency benchmarks.

## Prompt-only evaluation completion

- All1,000 tasks completed in2,682.55 seconds including process startup; measured evaluation time2,667.10 seconds. Peak allocated20,408,573,440 bytes; reserved21,604,859,904 bytes.
- Final Answer EM43.90%, F154.9473%, Joint Success36.60%, complete sentence evidence66.20%, document evidence81.30%. All1,000 trajectories made5 retrieval attempts and exhausted the action budget. Duplicate-call rate57.10%, JSON invalid0, no explicit stop. These belong to the new matched environment and do not replace the historical E0 baseline.
- A read-only AutoDL audit reconciled every episode checkpoint with the final JSONL, validated all1,000 hashes and the fixed task order/seed/action budget, verified all5,000 observation-only prompts, and reproduced every endpoint verifier result plus headline and behaviour aggregates exactly. Saved as `results/evaluation/prompt.integrity_audit.json`. All baseline artifacts and logs are mirrored locally.
- The final postprocessing auditor now includes the same per-evaluation reconciliation. This is additional evidence checking, with no change to scientific generation, evaluation or training source and no rerun of model outputs.
- At2026-09-05 13:44 UTC, E1 SFT evaluation786/1,000 tasks remained healthy. E2 evaluation and paired statistics follow automatically.

## E1 SFT evaluation completion

- All1,000 tasks completed in3,045.68 seconds including process startup, with3,027.07 seconds of measured evaluation. Peak allocated20,603,218,944 bytes; reserved22,737,321,984 bytes.
- Final Answer EM48.00%, F161.6061%, Joint Success43.20%, complete sentence evidence80.70%, document evidence89.20%; mean3.512 retrieval attempts. Retrieval histogram1:2,2:105,3:323,4:519,5:51. Explicit stops94.9%, budget exhaustion5.1%, premature stops17.8%, duplicate calls2.7572% per decision; JSON invalid and unsupported-tool rates0.
- All1,000 checkpoint hashes, final JSONL equality, fixed order/seed/budget and all4,461 observation-only prompts passed the independent audit. Endpoint verifier and all headline/behaviour metrics recomputed identically. Report SHA-256 `172917706f961076b698918ad917652173908c1866ffca57892dd84ea6d863c3`; trajectory SHA-256 `94dbbc2f4f507e9654c77b9665f538fa6f6d671f4ed1fa27ed3bd1f797009e64`. Reports, raw trajectories, checkpoints and logs are backed up locally.
- E2 GRPO evaluation began automatically and reached252/1,000 at2026-09-05 14:03 UTC, with a live process and fresh progress checkpoint. No scientific stage has been rerun.
- Added reporting-only scripts `summarise_adaptive_stopping.py` and `plot_adaptive_eval_behaviour.py`, to run after all three evaluations finish. They describe first next decisions after each retrieval count using post-episode gold labels, reconcile the existing post-two incomplete-evidence counts, retain all compact decision records and choose example cases by the first two lexicographic task IDs per category. Conditional evidence groups can differ across policies; their descriptive rates do not identify causal effects. No additional model calls or interventions are introduced.

## Final E2 evaluation and scientific audit

- E2 completed all1,000 tasks in1,985.25 stage seconds; measured evaluation1,962.41 seconds. Peak allocated20,531,869,184 bytes; reserved22,005,415,936 bytes. All13 driver stages completed, totalling25,556.88 seconds, without a scientific restart. The original paired comparison took10.59 seconds.
- E2 Answer EM45.6%, F157.9535%, Joint Success39.0%, complete sentence73.6%, document83.2%, mean retrieve2.215. Retrieval counts1:95,2:643,3:214,4:48; all1,000 explicitly stopped. Premature stop26.4%, duplicate0.6843% per decision, budget/JSON/unsupported0.
- Final integrity audit passed all3,000 evaluation checkpoint hashes, exact raw-output equality, fixed task order and all12,676 evaluation prompts; endpoint verification and both headline/behaviour aggregates recomputed identically. All three configurations match except adapter identity; data, retrieval, prompt and model-revision hashes match. Teacher candidate/selection/prompt checks also passed in the final audit. Scientific code since launch is unchanged apart from the separately documented diagnostic helper correction.
- Original paired bootstrap and supplemental ratio-of-sums behaviour bootstrap completed once each. E2 versus E1: EM−2.4pp [−4.5,−0.2], Joint Success−4.2pp [−6.5,−1.9], retrieval−1.297 [−1.344,−1.248], premature stops+8.6pp [6.2,10.9]. Negative outcomes are retained; no reward/seed adjustment or model rerun was made.
- Post-episode stopping audit produced9,676 compact decision records and reconciled the existing post-two incomplete-evidence counts. E2 stops140/241 times with incomplete evidence after two calls (58.09%); SFT23/265 (8.68%). Complete-evidence stop rates are503/664 (75.75%) and82/733 (11.19%) respectively. Different conditional groups are descriptive, not causal treatment/control cohorts.
- Both training and matched-evaluation figures were exported on AutoDL to PNG/SVG/CSV and visually checked locally. Labels, denominators and caveats are readable and unclipped. No scientific plots or metric calculations ran on Mac.
- E0 report checksums all passed again after completion. No Git commit/push, PDF change, extra GPU or scientific retry occurred. Final report and a separate three-bullet English CV draft preserve historical attribution and disclose the GRPO accuracy loss.
