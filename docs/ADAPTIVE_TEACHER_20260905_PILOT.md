# Frozen 7B adaptive-teacher pilot — 5 September 2026

The 50-task pilot completed on AutoDL. It supports continuing within the authorised 24-hour budget. It does **not** yet establish that adaptive stopping or downstream controller performance has improved. Full 1,617-task generation is in progress; new SFT, GRPO and benchmark evaluation results are pending.

## Fixed protocol and readiness

- One RTX 5090D; PyTorch 2.8.0+cu128 with sm_120 support. Real 7B generation and real Qwen3 NF4 QLoRA forward/backward/optimiser smoke checks passed.
- Teacher and finaliser use one frozen Qwen2.5-7B-Instruct model instance, revision `a09a35458c702b33eeacc393d103063234e8bc28`. All four required model snapshots passed content verification.
- First 50 tasks of the fixed 1,617-task training set, seed 42; at most two complete candidates per task and five actions per candidate. Endpoint clean-complete-evidence early stopping ended candidate sampling for nine tasks.
- Existing LanceDB dense + BM25 + RRF (k=60) + BGE reranker, 15 candidates and 6 returned sentences. Sentence IDs are retained; parent-child expansion is disabled.
- Gold labels were used for endpoint selection. An independent audit reproduced all 50 selections, checked 91 candidate hashes and verified that all 451 controller prompts equal the observation-only renderer. Retrieved corpus text can naturally contain an answer; answer-string matching is not a valid leakage test.

## Measurements

| Measurement | Selected candidate per task | All generated candidates |
|---|---:|---:|
| Trajectories | 50 | 91 |
| Answer EM | 64.00% | 60.44% |
| Answer F1 | 68.53% | 65.42% |
| Joint Success | 46.00% | 42.86% |
| Complete Sentence Evidence | 68.00% | 62.64% |
| Complete Document Evidence | 82.00% | 78.02% |
| Mean retrieval attempts | 4.200 | 4.275 |
| Legacy MeanToolCalls, including unsupported tool names | 4.500 | 4.637 |
| JSON parse/schema invalid rate, per decision | 0.00% | 0.00% |
| Unsupported tool action rate, per decision | 6.10% | 7.32% |
| Duplicate-call rate, per decision | 13.82% | 16.85% |
| Duplicate-call rate, per retrieval attempt | 16.19% | 19.54% |
| Premature explicit stop, per episode | 12.00% | 13.19% |
| Budget exhaustion | 58.00% | 68.13% |
| Explicit stop | 42.00% | 31.87% |

There were 246 selected decisions and 451 total decisions. JSON parse validity does not imply an executable valid action: 15 selected and 33 total decisions incorrectly named `final_answer` as a tool. The environment rejected these attempts. They are distinct from the valid stop action `{"final_answer": true}`.

| Retrieval attempts | Selected count / proportion | All-candidate count / proportion |
|---|---:|---:|
| 0 | 0 / 0% | 0 / 0% |
| 1 | 0 / 0% | 0 / 0% |
| 2 | 1 / 2% | 3 / 3.30% |
| 3 | 10 / 20% | 15 / 16.48% |
| 4+ | 39 / 78% | 73 / 80.22% |

The exact selected retrieval histogram was 2:1, 3:10, 4:17, 5:22. Decision lengths were four actions for 4 selected trajectories and five for 46; across all candidates, four actions for 4 and five for 87. Thus the pilot shows substantial dependence on the maximum action budget. A shift away from two calls alone is insufficient evidence of well-calibrated adaptive stopping.

After two retrievals, 23 selected trajectories still lacked complete gold sentence evidence. All 23 continued; seven subsequently completed the evidence set and four ended with Joint Success. Across all candidates the corresponding counts were 42 continued, eight completed evidence and five achieved Joint Success. These are post-episode diagnostic labels, never teacher inputs.

## Recovery and limits

- Selected trajectories contained 34 duplicate-call events. Six were followed by retrieval of new evidence (17.65%); none of those events ended in Joint Success. Across all candidates: 76 duplicate events, 13 followed by new evidence (17.11%) and two followed by both new evidence and endpoint Joint Success (2.63%). These rates use failure events as the denominator, including events at the last step; counts with a later-action opportunity were 23 and 52 respectively.
- No empty or failed retrieval events occurred in this pilot. Their recovery rates are **unresolved / no exposure**, not 0% or 100%.
- Contradictory-retrieval recovery is **unresolved**: this fixed protocol has no adjudicated contradiction labels. No synthetic perturbation task was introduced.
- These 50 training tasks were selected by the fixed task order, not a held-out benchmark. Best-of-N endpoint selection is expected to improve selected-candidate scores over raw candidates. Neither pilot scores nor that selection effect establish downstream model improvement.

## Time, memory and continuation decision

- Total measured in-process time: 275.85 s (4 min 36 s); 5.52 s/task including setup. Generation alone: 266.64 s, or 5.33 s/task. Mean candidate time: 2.93 s; mean candidates/task: 1.82.
- Peak GPU allocated memory: 17,024,435,712 bytes (17.02 GB). Peak reserved memory: 18,717,081,600 bytes (18.72 GB). These are PyTorch peaks, not a whole-device peak sampled by an external monitor.
- Full teacher estimate with the **maximum** four candidates for all 1,617 tasks: 5.26 h at the measured mean, 6.67 h using the measured candidate 90th percentile.
- Estimated remaining teacher + three 1,000-task evaluations + SFT/GRPO/startup: 9.71 h at the mean, 11.76 h using the candidate 90th percentile. The calculation reserves two hours for training and startup; those later stages are estimates, not measurements. It does not credit early candidate termination.
- Both estimates are below 24 hours. Full generation therefore started automatically with 1,617 tasks, four candidates maximum and five actions maximum. Seed, verifier, reward and data scope were not changed in response to pilot outcomes.

## Audit correction and artifacts

A diagnostic label initially conflated an unsupported tool named `final_answer` with a genuine stop. The postprocessing helper now separates them. The original pilot report is retained; `pilot50/integrity_audit.json` independently checks the candidate selections and supplies corrected selected action counts. The regression check passed on AutoDL (4 tests). No scientific trajectory was rerun for this reporting correction.

Local review copies are under `results/adaptive_teacher_20260905/pilot50/`: `report.json`, `rollouts.jsonl`, `rollouts.all-candidates.jsonl`, `selection.jsonl` and `integrity_audit.json`. Full per-candidate RNG/checkpoint records remain on AutoDL under `/root/autodl-tmp/e1-20260905/results/pilot50/rollouts.candidates/`. All model revisions and file SHA-256 values are in the mirrored `audit/autodl/model_download_result.json`.

E0 remains unchanged: its original 1,617 training trajectories all had exactly two retrievals and three decisions. The new pilot must not be used to relabel any E0 metric or checkpoint.
