# Qwen3-1.7B 多轮检索 Agent 后训练实验报告

> 实验日期：2026-08-05 至 2026-08-06
> 状态：SFT-v2、正式 GRPO、官方 validation 1,000 条同任务评测与配对统计均已完成。

## 1. 实验问题

在固定检索器、环境、最终答案生成器和评测集的前提下，比较以下四个 controller checkpoint：

1. Qwen3-1.7B prompt-only；
2. Qwen3-1.7B + QLoRA SFT-741；
3. Qwen3-1.7B + QLoRA SFT-v2；
4. Qwen3-1.7B + QLoRA SFT-v2 + GRPO。

关注两类结果：

- 端到端质量：Answer EM/F1、完整句级/文档级证据、Joint Success；
- Agent 行为与成本：非法动作、重复调用、预算耗尽、平均工具调用和 Calls / Success。

训练 reward 或训练集成功率不能替代固定 benchmark/dev 结果，更不能替代从未用于方案迭代的最终 test。

## 2. 开源数据与隔离方式

全部问题、答案、context 和 supporting facts 来自开源 HotpotQA distractor 数据，不自行编造。

| 用途 | 数据 | 规模 | 隔离方式 |
| --- | --- | ---: | --- |
| teacher / SFT / GRPO train | 官方 train 前 2,000 行 | 1,999 个有效任务 | 按全部可见 context 文档连通分量切为 1,617/191/191 |
| benchmark/dev | 官方 validation 前 1,000 行 | 1,000 个任务 | 与训练集隔离；固定复用，并用于失败诊断和方案迭代 |

train 原始窗口中有 1 条 supporting fact 句子编号越界，构建时显式跳过并记录原因。数据 manifest
保存源信息、切分规模和输入文件 SHA-256。Scripted teacher 只根据环境 observation 生成动作；
gold answer 和 supporting facts 只供 episode 结束后的 verifier 使用，不进入 controller prompt。
官方 validation 前 100 条也是这 1,000 条的诊断子集，因此不能把 1,000 条结果表述为 untouched final test。

## 3. SFT-v2

SFT-741 在固定 1,000 条 benchmark/dev 上消除了重复调用和预算耗尽，但完整句级证据从 41.3% 降到
35.0%，说明策略学会了停止，却经常停得过早。SFT-v2 针对这个失败模式修改数据选择和动作比例：

- 从 1,617 条官方 teacher 轨迹中保留 583 条 `CompleteSentenceEvidence=1` 的证据完整轨迹；
- 原始数据为 1,749 条决策：1,166 条工具动作、583 条停止动作；
- 只重复每条轨迹在停止前的最后一次工具动作；
- 最终为 2,915 条决策：2,332 条工具动作、583 条停止动作，工具/停止比例 4:1；
- messages 和 action 逐行对应原始 rollout，不新增或改写问题、答案与证据。

训练配置与结果：

| 项目 | 数值 |
| --- | ---: |
| Base model | Qwen3-1.7B |
| 方法 | QLoRA SFT |
| Examples | 2,915 |
| Epochs / optimizer steps | 2 / 366 |
| Truncated prompts | 0 |
| Train loss | 0.12492 |
| Runtime | 2,721.59 秒 |

## 4. 正式 GRPO

GRPO 从 SFT-v2 adapter 初始化。任务文件含 1,617 条训练任务，正式配置将可采样池限制为 128 条；
20 个 update 实际使用 40 个唯一任务组。每个 update 选择 2 个任务，每个任务生成 4 条同初始状态
rollout，使用组内相对 advantage；每批执行 2 个 optimization epochs。

训练实现包含：

- 只对 controller 生成的 action token 计算策略损失；
- 保存 old/current/reference token-level log-prob；
- clipped objective 与 reference KL；
- policy/reference 双 adapter，reference 冻结并做训练前后状态哈希；
- 对 loss、logits、梯度、参数和 optimizer state 做有限值检查；
- 记录 reward 分量、组内方差、KL、entropy、clip fraction、gradient norm 和 stop reason；
- 只保存 policy adapter，并独立重载完成推理验证。

正式配置的 reward 为：

```text
1.00 * task_success
+ 0.50 * evidence_coverage
- 0.02 * tool_call_count
- 0.20 * invalid_action
- 0.20 * duplicate_call
- 0.50 * premature_final_answer
- 0.30 * budget_exhausted
```

训练规模与资源：

| 项目 | 数值 |
| --- | ---: |
| Updates | 20 |
| Source / capped pool / unique task groups | 1,617 / 128 / 40 |
| Episodes / decisions | 160 / 525 |
| Tasks per update / rollouts per task | 2 / 4 |
| Optimization epochs | 2 |
| Runtime | 1,439.05 秒 |
| Total runtime | 1,458.58 秒 |
| Peak CUDA allocated | 约 9.73 GiB |
| Peak CUDA reserved | 约 10.22 GiB |
| Reference unchanged | 是，训练前后哈希一致 |
| Trained adapter SHA-256 | `6a5b14534c3e1cf0a07eda193d746f37447149de3af6198e717e17b193572e25` |

训练报告记录了官方训练数据、基础模型配置与全部权重分片、训练脚本、GRPO 模块、配置、初始
adapter、最终 adapter 的 SHA-256，以及 Python、Torch、Transformers、PEFT、bitsandbytes 和
Accelerate 的精确版本。

## 5. Reward hacking 压力测试

审计使用正式 GRPO 配置，而不是另一套默认权重。3/3 场景通过：

| 场景 | 结果 |
| --- | --- |
| 完整证据后增加无关检索 | coverage 保持 1.0，总回报 1.48 → 1.46 |
| 答案正确但证据只有 50% | `task_success=0`，总回报 -0.27 |
| 重复、失败、伪造参数或伪造引用 | coverage 不增加，并承担重复或调用成本 |

该审计证明当前检索/证据 reward 的已知刷分路径被拦截；它不代表所有开放式 reward 漏洞都已穷尽。

## 6. 官方 100 题快速诊断

三组运行使用相同官方 validation 前 100 条、同一 frozen finalizer、相同 seed 和 sampling 参数
`temperature=0.7, top_p=0.8, top_k=20`。

| Controller | Answer EM | Answer F1 | Joint Success | 完整句级证据 | 重复调用率 | 预算耗尽率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only | 21.0% | 26.94% | 13.0% | 38.0% | 63.00% | 100.0% | 5.00 |
| SFT-v2 | 20.0% | 27.42% | 12.0% | 43.0% | 11.39% | 1.0% | 3.75 |
| SFT-v2 + GRPO | 20.0% | 25.99% | 11.0% | 41.0% | 1.30% | 0.0% | 2.08 |

SFT-v2 在这 100 题上恢复了部分证据完整度。GRPO 相对 SFT-v2 将平均工具调用减少 44.5%，
并大幅压低重复调用，但 Answer 和 Joint Success 没有提升。这是方向诊断，不是最终主结果。

## 7. 官方 1,000 题固定 benchmark/dev 结果

四个运行的 1,000 个 task ID 和数据 manifest SHA-256 完全一致。所有区间均使用固定 seed 的
2,000 次逐题 paired percentile bootstrap。

| Controller | Answer EM | Answer F1 | Joint Success | 完整句级证据 | 完整文档证据 | 重复调用率 | 预算耗尽率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only | 18.80% | 25.31% | 11.70% | 41.30% | 61.10% | 63.68% | 100.00% | 4.998 |
| SFT-741 | 19.90% | 26.44% | 11.50% | 35.00% | 56.50% | 0.00% | 0.00% | 1.712 |
| SFT-v2 | 19.40% | 25.97% | 11.90% | 39.60% | 61.20% | 12.91% | 0.30% | 3.690 |
| SFT-v2 + GRPO | 19.90% | 26.16% | 11.90% | 37.40% | 59.20% | 1.23% | 0.00% | 2.096 |

Prompt-only → GRPO：

- 平均工具调用 `4.998 → 2.096`，减少 58.1%；逐题差值 -2.902，95% CI `[-2.920, -2.883]`；
- 重复调用率 `63.68% → 1.23%`，预算耗尽率 `100% → 0`；
- Answer EM `18.8% → 19.9%`，差值 +1.1pp，95% CI `[-0.8pp, +3.1pp]`；
- Joint Success `11.7% → 11.9%`，差值 +0.2pp，95% CI `[-1.6pp, +1.8pp]`；
- 完整句级证据 `41.3% → 37.4%`，差值 -3.9pp，95% CI `[-6.3pp, -1.4pp]`。

SFT-v2 → GRPO：

- 平均工具调用 `3.690 → 2.096`，减少 43.2%；逐题差值 -1.594，95% CI `[-1.629, -1.559]`；
- Joint Success 同为 11.9%，差值 95% CI `[-1.2pp, +1.2pp]`；
- Answer EM `19.4% → 19.9%`，差值 95% CI `[-0.9pp, +1.9pp]`；
- 完整句级/文档级证据分别下降 2.2pp/2.0pp，对应区间均不跨 0。

正式结论：GRPO 显著改变了行为效率，并在这一个 seed、这批重复使用的固定 benchmark/dev 上
基本保持 Answer/Joint 指标；没有证据支持准确率显著提升，而且证据完整度存在统计可见的下降。

## 8. 当前可写与不可写的结论

可以写：

- 构建了可复现、可验证的多轮 RAG Agent 后训练环境；
- 使用官方开源 HotpotQA 完成文档隔离的数据链路、QLoRA SFT 和真实单卡多轮 GRPO；
- 正式 GRPO 完成 160 个 rollout episode / 525 条决策和 20 次参数更新；
- reference 冻结、action-token loss、数值检查、reward 防刷审计、adapter 保存重载和全链路 provenance 均已验证；
- 固定 1,000 题 benchmark/dev 配对评测显示平均工具调用相对 prompt-only 减少 58.1%、预算耗尽降至 0，并基本保持 Answer/Joint 指标。

当前不能写：

- “GRPO 显著提升了问答准确率或 Joint Success”；
- “达到统计显著提升”；
- “多 seed 稳定优于 baseline”；
- “完成论文级训练与全部消融”。

小型机器可读报告随代码入库；adapter、完整轨迹与日志发布在
[GitHub Release](https://github.com/Chenypovo/Personal_RAG/releases/tag/agentic-rl-qwen3-1.7b-2026-08-06)。
