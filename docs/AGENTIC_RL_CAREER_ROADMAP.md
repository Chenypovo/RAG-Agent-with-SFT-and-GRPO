# Personal RAG 面向 Agent 算法岗的 Agentic RL 演进路线

> 更新时间：2026-08-04
> 目标：把当前的 Personal RAG + 长期记忆 Agent，从完整的 Agent 应用工程项目升级为能够支撑 Agent 算法岗面试的训练与实验项目。

## 1. 核心判断

当前项目已经覆盖了较完整的 Agent 系统能力：

- RAG、混合检索、reranker 和 parent-child 召回；
- 长期记忆的抽取、召回、冲突处理与非破坏更新；
- ReAct 风格的多步工具调用循环；
- 工具注册、失败重试、预算和死循环护栏；
- 基础轨迹记录，以及 baseline 与 ToolAgent 的评测框架。

因此，它目前适合证明“Agent 系统设计与工程落地”能力，但还不足以证明“Agent 算法/后训练”能力。主要缺口是：

- 没有被训练的开源 policy model；
- 没有 SFT、DPO、PPO 或 GRPO 训练过程；
- 没有面向多轮 Agent 的标准环境接口；
- 没有可学习、可分解、可防作弊的 reward；
- 没有训练曲线、对照实验和消融分析；
- 当前 LLM 通过 OpenAI-compatible API 返回文本，不能直接提供反向训练所需的 token IDs 和 logprobs。

如果目标是 Agent 算法岗，项目主线应从“继续堆 Agent 功能”切换成：

> 构建可验证的 Personal-RAG Agent 环境，并通过 SFT + GRPO 优化开源小模型的工具调用策略。

## 2. 建议的项目研究问题

把项目聚焦为一个可以通过实验回答的问题：

> 在冻结检索器、reranker、长期记忆规则和最终答案生成器的条件下，SFT 与 GRPO 能否提升小模型在查询拆解、工具选择、多跳证据收集、失败恢复和停止决策上的表现？

第一阶段只训练 controller policy：

- 冻结文档检索链路；
- 冻结长期记忆的存储与归并规则；
- 冻结最终答案生成器，并将 temperature 固定为 0；
- 训练模型决定调用什么工具、使用什么参数、如何根据观察继续行动，以及何时结束。

这样可以隔离变量，证明性能变化来自 Agent 决策策略，而不是更换了检索器或生成模型。

## 3. 从现有代码到 RL 闭环的映射

| RL 概念 | 当前项目中的基础 | 需要补充的能力 |
| --- | --- | --- |
| State | 用户问题、history、plan、已执行步骤和工具观察 | 标准化、可序列化、可复现的 observation |
| Action | `retrieve_docs`、`read_memory`、`write_memory`、`calculator`、`final_answer` | 严格 action schema，以及 policy 生成的 token/logprob |
| Transition | `ToolRegistry.dispatch()` | 独立的 `reset()` / `step()` 环境接口 |
| Episode | `ToolAgent.chat()` 控制循环 | task ID、seed、环境版本和状态快照 |
| Reward | task success、coverage、tool precision/recall | 结果奖励、过程 shaping、安全惩罚和 reward 明细 |
| Trajectory | `TrajectoryStep` | 原始 messages、action tokens、logprobs、reward、done 和 policy version |

建议新增独立模块，而不是把训练逻辑直接塞进线上 `ToolAgent`：

```text
app/agent_rl/
├── env.py              # PersonalRAGEnv: reset / step
├── tasks.py            # 任务加载、切分和环境初始化
├── rewards.py          # 可验证 reward 与 reward breakdown
├── rollout.py          # 批量、多样本 rollout
├── trajectory.py       # 训练所需 episode/transition schema
├── verifier.py         # 答案、证据、工具和记忆安全验证
└── adapters.py         # 复用现有 ToolRegistry/RAG/memory 的适配层

scripts/
├── build_agent_sft_data.py
├── train_agent_sft.py
├── train_agent_grpo.py
└── eval_agent_policy.py
```

## 4. 第一阶段：构建可复现 Agent Environment

环境建议采用接近 Gymnasium 的接口：

```python
env = PersonalRAGEnv(task=task, runtime=runtime, seed=42)

observation = env.reset()
observation, reward, terminated, info = env.step(action)
```

### Observation

至少包含：

- 当前用户任务；
- 可用工具及参数 schema；
- 已执行的 action 与对应 observation；
- 当前已收集证据；
- 当前计划或待办；
- 剩余 step/tool budget；
- 环境允许暴露给 policy 的错误信息。

### Action

使用严格的结构化动作：

```json
{"tool": "retrieve_docs", "args": {"query": "..."}}
```

或：

```json
{"final_answer": true}
```

训练阶段不建议继续依赖可选的自由文本 `thought`。它会增加动作空间、训练 token 数和评测歧义。需要解释时，可保留简短 rationale，但不应作为环境执行条件。

### Episode 隔离

每个 episode 都必须：

- 使用固定版本的文档索引；
- 使用独立的 memory store 快照；
- 固定随机 seed；
- 固定检索参数、prompt version 和工具版本；
- 结束后丢弃训练产生的记忆写入；
- 保存完整 trajectory 和 reward breakdown。

`write_memory` 尤其需要 sandbox，不能让一次训练 rollout 修改另一次 rollout 或生产数据。

## 5. 第二阶段：任务与训练数据

当前少量 example task 只能用于冒烟，不能支撑训练和可信评测。建议先构造 500～2000 条任务，再根据算力和效果扩展。

任务至少覆盖：

1. 无需工具即可回答；
2. 单跳文档检索；
3. 两跳或三跳文档检索；
4. 查询改写与检索失败后的重试；
5. 长期记忆读取；
6. 应该写入长期记忆；
7. 不应该写入长期记忆；
8. 记忆冲突、更新和撤销；
9. 检索数值后调用 calculator；
10. 工具返回为空、参数错误或内部异常后的恢复；
11. 文档 prompt injection 与越权写记忆；
12. 有限预算下的停止决策。

### 数据切分

必须按文档来源或知识主题切分 train/dev/test，不能把同一文档生成的改写问题随机分散到三个集合。否则会产生知识泄漏，无法证明 Agent 对新文档、新问题的泛化能力。

### SFT 数据

使用较强 teacher model 为每个任务生成多条轨迹，然后通过程序化 verifier 筛选成功轨迹。保留：

- 完整多轮 messages；
- 每次 tool call；
- 工具 observation；
- 最终答案；
- verifier 的各项结果；
- 生成模型、prompt、temperature 和 seed。

先训练并比较：

1. 原始 instruct model；
2. prompt-only ToolAgent；
3. tool-use SFT model。

只有当 SFT 已能稳定输出合法 action 并完成基础工具调用时，才进入 RL 阶段。

## 6. 第三阶段：GRPO Reward 设计

初版 reward 应以可验证结果为主，以小幅过程奖励辅助学习。例如：

```text
R =
  1.00 * task_success
+ 0.30 * evidence_coverage
+ 0.10 * citation_correctness
+ 0.05 * valid_action_format
- 0.02 * tool_call_count
- 0.10 * invalid_or_duplicate_call
- 0.30 * premature_final_answer
- 1.00 * unsafe_memory_write
```

设计原则：

- task success 必须是主要信号；
- 不要重奖“调用了预期工具”，否则模型会无脑调用全部工具；
- 工具调用成本只做轻量惩罚，避免模型为了省调用而提前结束；
- memory 的误写、越权写和破坏性更新应给予高额负奖励；
- 开放式任务可以拆成带证据的 checklist，由多个二元 verifier 分别判定；
- 训练报告必须展示每个 reward component，不能只展示总 reward。

推荐训练顺序：

```text
Instruct model
    → successful teacher trajectories
    → LoRA/QLoRA SFT
    → on-policy multi-turn rollout
    → GRPO
    → held-out evaluation + ablation
```

初次打通可使用 Hugging Face TRL 的 `GRPOTrainer` 和 `environment_factory`。如果后续需要多 GPU、高吞吐、异步工具调用和 rollout/training 分离，再迁移到 verl + vLLM/SGLang。

## 7. 当前代码在训练前需要修正的问题

### 7.1 Calculator 结果未进入最终合成

当前 `_accumulate()` 只归集文档 chunks、memories 和 memory ops，没有保存 calculator result；最终 generator 也只接收 chunks 和 memory。因此 controller 即使正确计算，最终回答阶段也可能看不到计算结果。

应把所有工具 observation 纳入正式 episode state，并让最终回答能够消费结构化工具结果。

### 7.2 当前 verifier 容易被 reward hacking

现有程序化判定中：

- multihop 主要检查 gold chunk coverage；
- calc 主要检查答案中是否出现预期数字；
- memory 主要检查字符串是否包含关键词。

RL policy 会主动寻找这些规则的漏洞，因此需要同时验证：

- 最终答案是否正确；
- 关键结论是否被证据支持；
- 引用是否真实对应证据；
- calculator 输出是否被正确使用；
- 是否存在无意义或重复工具调用；
- 是否发生不必要、错误或敏感的记忆写入。

### 7.3 当前 LLM 接口不能直接训练

现有 `complete_fn` 只返回 API 文本。实际 RL 需要访问模型权重，并在 rollout 时保存 policy 原始生成的 token IDs、token-level logprobs 和 mask。

因此：

- OpenAI-compatible API 可继续用于 teacher、数据合成或 judge；
- 被训练的 controller 必须改为可控的开源模型；
- rollout 可使用 Transformers、vLLM 或 SGLang；
- 参数更新可使用 TRL、verl 或其它支持 GRPO/PPO 的训练框架。

## 8. 必须完成的实验矩阵

主结果至少包含：

| 方法 | Task Success | Multi-hop Coverage | Tool F1 | Invalid Call Rate | Calls / Success | Unsafe Memory Write Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| SFT | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |
| SFT + GRPO | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 |

至少完成三组消融：

- outcome-only reward vs shaped reward；
- SFT-only vs SFT + GRPO；
- 有无工具成本惩罚；
- 不同 max steps/tool budget；
- 不同 group rollout 数量；
- 有无失败恢复训练任务；
- 不同数据规模；
- 冻结 final generator vs controller 与 final answer 联合训练。

训练期间还应监控：

- reward mean/std；
- 同一 prompt 的组内 reward variance；
- policy entropy；
- KL divergence；
- gradient norm；
- action 长度；
- 重复工具调用率；
- 每种 stop reason 的比例；
- train 与 held-out success 的差距。

不能只展示训练 reward 上升。多轮 Agent RL 可能出现策略熵下降、重复轨迹、reward collapse，以及 reward 上升但真实成功率下降。

## 9. 面试需要能够解释的问题

完成项目后，应能清楚回答：

1. 这个 Agent 环境的 state、action、transition 和 reward 分别是什么？
2. 为什么该问题需要 RL，SFT 为什么不够？
3. 为什么选择 GRPO，而不是 DPO、PPO 或纯 rejection sampling？
4. episode-level reward 如何分配给多轮 action token？
5. 工具 observation 是否参与 loss，如何做 loss mask？
6. 如何防止 reward hacking？
7. 为什么每个 rollout 必须使用独立 memory snapshot？
8. 如何保证同组 GRPO rollout 具有相同初始状态？
9. 为什么 train/dev/test 要按文档来源切分？
10. 如何证明提升来自 RL，而不是更多数据、更强 teacher 或更强 base model？
11. reward 上升但 held-out success 下降时如何定位？
12. 如何监控并缓解重复调用、策略熵坍缩和训练不稳定？

## 10. 简历表达模板

真实实验完成后，可以写成：

> 构建可复现的 Personal-RAG Agent 训练环境，将多跳检索、长期记忆和计算工具建模为多轮决策过程；构造 X 条可验证任务及基于答案正确性、证据覆盖、工具成本和记忆安全的组合奖励，基于 Qwen-XB 完成 LoRA SFT 与 GRPO 后训练，使 held-out 任务成功率由 X% 提升至 Y%，非法工具调用率下降 Z%，并通过 reward、步数预算和数据规模消融分析长程训练稳定性。

没有真实数字前，不应在简历中填写占位结果。最终交付物应包括：

- 可运行的环境与训练代码；
- 版本化的 train/dev/test 数据；
- SFT 和 GRPO 配置；
- 可复现实验命令；
- 主结果表、训练曲线和消融实验；
- 失败案例与 reward hacking 分析；
- 对算力、成本和方法边界的诚实说明。

## 11. 优先级与取舍

建议按以下顺序推进：

### P0：先让评测可信

- 修正 calculator 结果丢失；
- 扩展 verifier；
- 建立严格 train/dev/test 切分；
- 生成 prompt-only baseline 数字；
- 实现 episode 隔离和可复现轨迹。

### P1：完成训练闭环

- 接入开源小模型；
- 构造并筛选 teacher trajectories；
- 完成 LoRA/QLoRA SFT；
- 接入 GRPO rollout、reward 和参数更新；
- 跑通 Prompt-only、SFT、SFT + GRPO 三组对照。

### P2：做成算法项目

- 完成 reward、数据量、步数预算等消融；
- 分析失败轨迹和训练稳定性；
- 输出完整技术报告；
- 将核心环境与数据构造过程做成可复现开源模块。

短期内不建议优先投入：

- 再增加大量普通工具；
- 为展示而引入 multi-agent；
- 继续扩展 Web UI；
- 只跑通一个 GRPO 命令但没有可信 verifier 和对照实验；
- 使用闭源 API 生成几条轨迹后直接宣称完成 Agentic RL。

## 12. 最终定位

本项目最有竞争力的方向不是“大而全的个人助手”，而是：

> 一个真实 RAG + 长期记忆系统上的可验证 Agentic RL 研究平台，用于训练和评估模型的工具选择、多跳检索、失败恢复与安全记忆决策。

当项目能够提供训练代码、真实曲线、对照实验、消融分析和失败案例时，它才从 Agent 工程项目升级为能够支撑 Agent 算法岗面试的算法项目。

## 参考资料

- [Hugging Face TRL：GRPO Agent Training](https://huggingface.co/docs/trl/main/grpo_trainer)
- [verl：Agentic RL Training](https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst)
- [Agent Lightning](https://microsoft.github.io/agent-lightning/latest/)
- [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073)
- [CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use](https://arxiv.org/abs/2602.12268)
