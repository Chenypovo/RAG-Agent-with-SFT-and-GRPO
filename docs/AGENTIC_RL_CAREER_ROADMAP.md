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

- 已完成 Qwen3-1.7B 的 75 条 decision QLoRA SFT 小样本训练，但 held-out 效果退化，尚无有效的后训练 policy；
- 尚未完成扩大数据后的 SFT，也没有 DPO、PPO 或 GRPO 训练结果；
- 已有最小多轮环境接口，但尚未接入 policy token、logprob 和批量 rollout；
- 已有第一版可分解 reward/verifier，但还需要防作弊压力测试；
- 缺少足够规模的训练曲线、多 seed 对照和消融分析；
- 已接入本地 Transformers + PEFT 完成 SFT；GRPO 所需的 on-policy token、logprob 和批量 rollout 仍未接通。

如果目标是 Agent 算法岗，项目主线应从“继续堆 Agent 功能”切换成：

> 构建可验证的 Personal-RAG Agent 环境，并通过 SFT + GRPO 优化开源小模型的工具调用策略。

### 1.1 已落地的无 GPU 检查点

当前分支已经完成训练前的最小闭环：

- `PersonalRAGEnv.reset()/step()`、严格 action schema、step budget、重复调用惩罚和可序列化轨迹；
- HotpotQA distractor 数据转换，200 个任务、8171 条句级语料，以及按全部可见 context 文档连通分量隔离的 162/18/20 划分；
- HotpotQA 风格 Answer EM/F1、句级/文档级证据 verifier；
- 不读取金标答案的确定性双轮检索 controller，并保存可复现 action trajectory；
- 相关环境、数据、verifier 和 controller 测试全部通过。

2026-08-04 已按全部可见 context 文档连通分量重新构建 smoke 数据，并复现 all partition 的
BM25 与 scripted 结果。新划分为 train/validation/test = 162/18/20；同一可见 context 文档
关联的任务不会跨集合。数据本体因体积不入库，但 manifest 和结果报告已跟踪：

- `data/agent_rl/manifests/hotpotqa_smoke.json`；
- `data/agent_rl/reports/hotpotqa_smoke_bm25.json`；
- `data/agent_rl/reports/hotpotqa_smoke_scripted.json`。

另外已准备 1,000 条官方 HotpotQA validation 作为纯 test held-out，包含 40,330 条句级语料；
其 manifest 为 `data/agent_rl/manifests/hotpotqa_validation_1k.json`。这批数据不再参与内部
train/validation 哈希切分，避免将官方 held-out 重新混回训练集合。

在相同的 8 条检索预算下，200 条 smoke set 的检索结果为：

| Controller | 预算分配 | Sentence Recall | 完整句级证据 | Document Recall | 完整文档证据 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单轮 BM25 | `k=8` | 65.69% | 36.00% | 75.75% | 53.00% |
| Scripted two-hop | `k=7 + k=1` | 66.78% | 38.00% | 76.75% | 55.50% |

这只是用于验证环境和查询改写方向的 smoke baseline，不是最终论文级结果。当前 finalizer 主动 abstain，因此 Answer EM/F1 和 Joint Success 均为 0；不能把上述证据召回提升表述成端到端问答提升。smoke 内部 validation 只有 18 条、test 只有 20 条，也不足以支撑显著性结论；可信度判断以下面的官方 validation 1,000 条纯 test held-out 为准。

官方 validation 1,000 条纯 test held-out 已完成 BM25 与 scripted two-hop 评测。数据包含
40,330 条句级语料；两种方法都使用 8 条检索结果，避免用更多召回条数制造表面增益：

| Controller | 预算分配 | Sentence Recall | 完整句级证据 | Document Recall | 完整文档证据 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单轮 BM25 | `k=8` | 64.62% | 34.70% | 75.80% | 55.00% |
| Scripted two-hop | `k=7 + k=1` | 64.92% | 35.70% | 75.65% | 55.10% |
| Scripted - BM25 | — | +0.31 个百分点 | +1.00 个百分点 | -0.15 个百分点 | +0.10 个百分点 |

Scripted 的句级 Precision/F1 为 20.79% / 31.12%，文档级 Precision/F1 为
27.97% / 40.24%。这组结果的提升很小，而且 Document Recall 略有下降；它只能证明
两跳查询改写和轨迹评测链路已经跑通，不能据此宣称 scripted controller 稳定优于 BM25。
由于这一阶段没有答案生成器，策略会主动拒答，Answer EM/F1、Joint EM/F1 和
Joint Success 都是 0。上述数字不是端到端问答效果，也不是 SFT 或 GRPO 效果。

对应的可机读报告为：

- `data/agent_rl/reports/hotpotqa_validation_1k_bm25.json`；
- `data/agent_rl/reports/hotpotqa_validation_1k_scripted.json`。

报告记录了输入文件 SHA-256、manifest SHA-256、Python 与 `rank-bm25` 版本；仓库内
`data/agent_rl/manifests/hotpotqa_validation_1k.json` 是运行目录 `manifest.json` 的同哈希副本。

复现命令：

```bash
python scripts/prepare_hotpotqa_agent_rl.py --limit 200 --output-dir data/agent_rl/hotpotqa_smoke
python scripts/eval_hotpotqa_bm25.py --data-dir data/agent_rl/hotpotqa_smoke --partition all --ks 4,8,20
python scripts/eval_hotpotqa_scripted_agent.py --data-dir data/agent_rl/hotpotqa_smoke --partition all

# 官方 validation 1k 纯 test held-out；数据需先按 tracked manifest 恢复到对应目录
python scripts/eval_hotpotqa_bm25.py \
  --data-dir data/agent_rl/hotpotqa_validation_1k --partition test --ks 4,8,20 \
  --output data/agent_rl/reports/hotpotqa_validation_1k_bm25.json --overwrite
python scripts/eval_hotpotqa_scripted_agent.py \
  --data-dir data/agent_rl/hotpotqa_validation_1k --partition test \
  --output data/agent_rl/reports/hotpotqa_validation_1k_scripted.json \
  --trajectories data/agent_rl/hotpotqa_validation_1k/scripted_trajectories_test.jsonl --overwrite
```

### 1.2 Prompt-only 端到端评测检查点

当前分支已补齐不依赖训练的端到端评测代码：

- 严格 JSON 的 `PromptOnlyPolicy`，非法输出不做静默修复，而是交给环境记录和惩罚；
- 与 controller 分离的冻结答案生成器，只接收问题和工具观察，不读取金标答案；
- 通用 policy rollout，保存模型原始输出、完整 prompt、解析错误、环境 transition、分项 reward、seed 和版本；
- 同时报 Answer EM/F1、证据覆盖、Joint Success、非法动作率、重复调用率、平均工具调用数、预算终止率和拒答率；
- 数据 manifest 哈希、controller/finalizer 模型名、温度、prompt 版本和环境版本进入实验报告。

评测入口：

```bash
python scripts/eval_hotpotqa_prompt_policy.py \
  --data-dir data/agent_rl/hotpotqa_smoke \
  --partition test \
  --env-file /path/to/.env \
  --controller-model <model> \
  --finalizer-model <model>
```

GPU 机器也可使用 `--completion-backend transformers` 直接加载同一个开源模型；controller 与
finalizer 共享权重但使用独立 prompt，并关闭 thinking mode。这样无需部署额外 API 服务，也避免
同一模型在显存中重复加载。

Qwen3-1.7B controller 与同模型 frozen finalizer 已在官方 validation 的固定 100 条 held-out
任务上完成 prompt-only 和 SFT 对照。两组 sampling 实验均使用
`temperature=0.7, top_p=0.8, top_k=20`，greedy SFT 使用 `temperature=0`：

| Controller | Answer EM | Answer F1 | Joint Success | 完整句级证据 | 重复调用率 | 预算终止率 | 非法动作率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only，sampling | 21.00% | 26.94% | 13.00% | 38.00% | 63.00% | 100.00% | 0.00% | 5.00 |
| QLoRA SFT-75，sampling | 12.00% | 16.96% | 4.00% | 11.00% | 1.32% | 0.00% | 12.33% | 1.07 |
| QLoRA SFT-75，greedy | 13.00% | 18.73% | 4.00% | 16.00% | 1.36% | 5.00% | 13.64% | 0.98 |

Prompt-only 基线暴露出两个明确问题：每题都跑满 5 次工具调用，预算终止率 100%，重复调用率
63%。它仍取得 Answer EM 21%、Answer F1 26.94%、Joint Success 13%，因此是当前端到端基线，
不是可直接部署的策略。

报告已跟踪：

- `data/agent_rl/reports/qwen3_1.7b_prompt_validation100.json`；
- `data/agent_rl/reports/qwen3_1.7b_sft75_validation100.json`；
- `data/agent_rl/reports/qwen3_1.7b_sft75_validation100_greedy.json`。

### 1.3 QLoRA SFT 小样本诊断

Teacher 在 162 个 train episodes 上生成 486 条 decision，其中 25 个 episode 达到
`JointSuccess=1`。严格筛选成功且无解析错误的轨迹后，得到 75 条 controller decision；QLoRA
训练使用 2 epochs / 10 steps，用时 60.37 秒，最终 train loss 为 1.261。

这次小数据 SFT 没有带来端到端提升。Sampling 下，平均工具调用从 5.00 降到 1.07、预算终止率
从 100% 降到 0%，但 Answer EM 从 21% 降到 12%、Joint Success 从 13% 降到 4%、完整句级
证据从 38% 降到 11%，并新增 12.33% 非法动作。Greedy 复验仍只有 13% Answer EM、4%
Joint Success 和 16% 完整句级证据，说明退化不是 sampling 波动能够解释的。

结论是：只用 25 个成功 episode / 75 条 decision 训练，使 policy 倾向过早停止，且数据不足以
稳定学习严格动作格式。本轮价值在于发现失败模式，不能写成“SFT 提升”。下一轮必须扩大 teacher
轨迹，加入不同长度、失败恢复和 hard-negative 决策，并先证明 SFT 在 held-out 上不退化，再进入
GRPO。当前没有 GRPO 训练结果。

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
├── policies.py         # prompt-only controller 与严格动作解析
├── finalizers.py       # 冻结、不可见金标答案的终局合成
├── rollouts.py         # policy 决策、transition 与版本化轨迹
├── scripted_agent.py   # 无 GPU 的确定性 controller 与 rollout
├── trajectory.py       # 训练所需 episode/transition schema
├── verifiers.py        # 答案与句级/文档级证据验证
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

当前已完成的 100 条 held-out 阶段结果为：

| 方法 | Joint Success | Answer F1 | 完整句级证据 | Invalid Call Rate | Mean Tool Calls | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prompt-only sampling | 13.00% | 26.94% | 38.00% | 0.00% | 5.00 | 当前基线，重复调用和预算耗尽严重 |
| SFT-75 sampling | 4.00% | 16.96% | 11.00% | 12.33% | 1.07 | 小数据导致过早停止，端到端退化 |
| SFT-75 greedy | 4.00% | 18.73% | 16.00% | 13.64% | 0.98 | 退化在 greedy 下仍存在 |
| SFT + GRPO | 未完成 | 未完成 | 未完成 | 未完成 | 未完成 | 不得写入简历成果 |

正式主结果仍需在更大 held-out、多 seed 下补齐 Tool F1、Calls / Success、Unsafe Memory Write Rate，
并完成 SFT + GRPO 后才可形成算法提升结论。

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

当前可写的真实版本见 `docs/RESUME_AGENTIC_RL_CN.md`。本轮应写“完成环境、基线、QLoRA
小样本试验并定位过早停止”，不能写“SFT 或 GRPO 提升了效果”。

扩大数据并完成真实实验后，才可以升级成：

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

- 已完成：修正 calculator 结果丢失；
- 部分完成：答案与句级/文档级证据 verifier 已落地，reward hacking 压力测试仍待做；
- 已完成：按全部可见 context 文档连通分量建立 train/validation/test 切分，并重建 smoke 数据与报告；
- 已完成：准备并评测 1,000 条官方 validation 纯 test held-out，保存内容哈希 manifest 与 BM25/scripted 报告；
- 已完成：AutoDL 上的 Qwen3-1.7B frozen finalizer + prompt-only controller 100 条 held-out 完整指标报告；
- 部分完成：环境状态 reset、seed 和可复现轨迹已落地，memory store 独立快照仍待做。

### P1：完成训练闭环

- 已完成小规模：接入 Qwen3-1.7B，并从 162 个 teacher episodes 筛选 25 个成功 episode；
- 已完成诊断但效果退化：75 条 decision 的 QLoRA SFT 与 sampling/greedy held-out 对照；
- 进行中：已准备 1,999 个有效任务的官方 train 扩展集（`1617/191/191`），正在生成 teacher trajectories 并训练扩大版 SFT；
- 待验证：扩大版 SFT 在官方 1,000 条 held-out 上的指标至少不低于 prompt-only；
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
