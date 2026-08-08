# Agentic RAG / Agent RL 对齐检索栈与 7B Finalizer 实验

> 实验日期：2026-08-06
> 状态：Hybrid、reranker top-k 消融、四 controller 的 1,000 题评测及新旧栈配对统计均已完成。

## 1. 改动

Agentic RAG 默认向量后端是 LanceDB，不是 ChromaDB。本轮将 Agent RL 的评测检索链路对齐为同一套
`LanceDB + BM25 + RRF + cross-encoder reranker` 组件：

- Dense encoder：`BAAI/bge-small-en-v1.5`，normalized CLS pooling；
- 向量库：LanceDB；稀疏检索：仓库现有 `BM25Store`；
- 两路各粗筛 top-15，RRF `k=60` 后保留 15 个候选；
- Reranker：`BAAI/bge-reranker-base`；
- Reranker 输出 top-6；
- Frozen finalizer：`Qwen/Qwen2.5-7B-Instruct`，BF16；
- Controller 仍为原 Qwen3-1.7B prompt/SFT/GRPO checkpoint。

Agent RL 保留句级 chunk，且不启用 parent-child expansion，避免扩展文本与
`source#chunk_id` 证据记账不一致。训练和 validation 分别构建了 77,864 / 40,330 条
句级 LanceDB 记录。

## 2. Reranker top-k 选择

以下结果均为同一批官方 HotpotQA validation 1,000 题的一次完整问题检索：

| Retriever | 返回 k | 完整句证据 | Sentence Recall | 完整文档证据 | Document Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 8 | 34.7% | 64.62% | 55.0% | 75.8% |
| Hybrid | 6 | 46.8% | 73.34% | 66.7% | 82.9% |
| Hybrid | 8 | 53.2% | 77.17% | 72.4% | 86.0% |
| Hybrid + reranker | 6 | **59.5%** | **80.83%** | **77.6%** | **88.7%** |
| Hybrid + reranker | 8 | 64.0% | 83.17% | 81.3% | 90.6% |

选择 top-6：它比未重排 Hybrid@8 少返回 25% chunk，但完整句证据高 6.3pp；也高于原
BM25@20 的 55.2%。top-8 召回更高，但会增加每次 observation 和 finalizer 上下文成本。

## 3. 新栈四 controller 结果

固定相同 hybrid/reranker/7B finalizer、官方 validation 1,000 题、sampling seed 42：

| Controller | Answer EM | Answer F1 | Joint Success | 完整句证据 | 完整文档证据 | 重复调用率 | 预算耗尽率 | 平均调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only | **44.2%** | **55.31%** | 36.1% | 64.2% | 79.4% | 61.78% | 100.0% | 5.000 |
| SFT-741 | 43.3% | 54.02% | 33.5% | 62.1% | 81.8% | 0.00% | 0.0% | 1.722 |
| SFT-v2 | 43.6% | 54.56% | **36.2%** | **67.7%** | **85.0%** | 11.42% | 0.6% | 3.726 |
| SFT-v2 + GRPO | 42.5% | 53.60% | 35.0% | 65.5% | 83.6% | **1.28%** | **0.0%** | 2.121 |

在新栈内，Prompt-only → GRPO：

- 平均工具调用 `5.000 → 2.121`，减少 57.6%，95% CI `[-2.899, -2.858]`；
- 重复调用率 `61.78% → 1.28%`，预算耗尽率 `100% → 0`；
- Answer EM -1.7pp，95% CI `[-4.2pp, +0.8pp]`，不显著；
- Joint Success -1.1pp，95% CI `[-3.5pp, +1.5pp]`，不显著；
- 完整句证据 +1.3pp，95% CI `[-1.1pp, +3.7pp]`，不显著；
- 完整文档证据 +4.2pp，95% CI `[+2.2pp, +6.1pp]`。

SFT-v2 → GRPO 仍存在效率/证据 trade-off：平均调用减少 43.1%，但完整句证据下降
2.2pp，95% CI `[-3.5pp, -1.1pp]`；Answer EM 和 Joint Success 的差异区间仍跨 0。

## 4. 旧栈与新栈

旧栈为 BM25@8 + Qwen3-1.7B finalizer；新栈同时升级检索、重排和 finalizer。逐题 paired
bootstrap（2,000 次）结果：

| Controller | Answer EM | Joint Success | 完整句证据 |
| --- | ---: | ---: | ---: |
| Prompt old → new | 18.8% → 44.2% (+25.4pp) | 11.7% → 36.1% (+24.4pp) | 41.3% → 64.2% (+22.9pp) |
| SFT-741 old → new | 19.9% → 43.3% (+23.4pp) | 11.5% → 33.5% (+22.0pp) | 35.0% → 62.1% (+27.1pp) |
| SFT-v2 old → new | 19.4% → 43.6% (+24.2pp) | 11.9% → 36.2% (+24.3pp) | 39.6% → 67.7% (+28.1pp) |
| GRPO old → new | 19.9% → 42.5% (+22.6pp) | 11.9% → 35.0% (+23.1pp) | 37.4% → 65.5% (+28.1pp) |

GRPO 新旧差值的 95% CI：

- Answer EM：`[+19.6pp, +25.5pp]`；
- Joint Success：`[+20.1pp, +26.0pp]`；
- 完整句证据：`[+25.2pp, +31.3pp]`。

这些区间证明**组合系统升级**有效，但不能把增益单独归因给 7B finalizer、hybrid 或
reranker。检索消融只直接证明证据召回提升；没有完成 1,000 题 finalizer-only 消融。

## 5. 可写与不可写

可以写：

- 将 Agentic RAG 与 Agent RL 对齐到共享的 LanceDB/BM25/RRF/BGE reranker 检索组件；
- 通过 1,000 题检索消融选择 `15 → 6` 的候选/重排配置；
- 组合栈将 GRPO 的 Answer EM 从 19.9% 提升到 42.5%、Joint Success 从 11.9% 提升到
  35.0%，对应配对区间均不跨 0；
- 在新栈上，GRPO 相比 prompt-only 减少 57.6% 工具调用并消除预算耗尽，同时
  Answer EM/Joint Success 差异不显著。

不能写：

- “GRPO 使 Answer EM 从 19.9% 提升到 42.5%”；
- “7B finalizer 单独贡献了全部准确率提升”；
- “在 untouched test 或多 seed 上稳定提升”；
- “现有 GRPO 已在 hybrid 环境重新训练”。

当前 adapter 仍是在 BM25 observation 分布上训练，本轮只在对齐后的新检索栈上重新评测。
下一步若要研究训练收益，应使用已构建的 train LanceDB 索引重新生成 teacher/SFT 轨迹并重训 GRPO。
