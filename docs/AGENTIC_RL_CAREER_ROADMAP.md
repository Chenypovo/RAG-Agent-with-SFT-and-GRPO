# Personal RAG 面向 Agent 算法岗的 Agentic RL 演进路线

> 更新时间：2026-08-06
> 目标：把当前的 Personal RAG + 长期记忆 Agent，从完整的 Agent 应用工程项目升级为能够支撑 Agent 算法岗面试的训练与实验项目。

## 1. 核心判断

当前项目已经覆盖了较完整的 Agent 系统能力：

- RAG、混合检索、reranker 和 parent-child 召回；
- 长期记忆的抽取、召回、冲突处理与非破坏更新；
- ReAct 风格的多步工具调用循环；
- 工具注册、失败重试、预算和死循环护栏；
- 基础轨迹记录，以及 baseline 与 ToolAgent 的评测框架。

当前分支已经从纯 Agent 应用工程进入可运行的后训练阶段，但正式结论仍受单模型、单 seed 和
缺少消融限制。当前状态是：

- 已完成 Qwen3-1.7B 的 75 条 decision 小样本诊断和 741 条 decision 扩大版 QLoRA SFT；
- 扩大版 SFT 修复了重复调用和预算耗尽，但固定 1,000 条 benchmark/dev 的 Joint Success 未提升，尚无端到端效果更强的后训练 policy；
- 已从 583 条完整句级证据 teacher 轨迹构建 SFT-v2，共 2,915 条可追溯 controller decision；
- 已完成 20-update 的正式 GRPO 训练：160 个同任务分组 rollout、525 条多轮决策，并保存可重载 policy adapter；
- 已接通生成 action token 的 old/current/reference log-prob、组内相对 advantage、clipped objective、KL 和两轮参数更新；
- 正式奖励配置已通过 3/3 防刷压力测试，重复、失败和伪造证据均不能增加 coverage；
- 已完成 Prompt-only、SFT-741、SFT-v2 与 SFT-v2 + GRPO 的官方 validation 1,000 条同任务对照；
- 缺少足够规模的训练曲线、多 seed 对照和消融分析；
- 当前正式 GRPO 规模用于验证训练闭环与首轮策略效果，不等同于论文级充分训练。

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

另外已准备 1,000 条官方 HotpotQA validation 作为固定 benchmark/dev，包含 40,330 条句级语料；
其 manifest 为 `data/agent_rl/manifests/hotpotqa_validation_1k.json`。这批数据与训练集隔离，不参与
内部 train/validation 哈希切分；但它已被重复用于失败诊断和方案迭代，不是 untouched final test。

在相同的 8 条检索预算下，200 条 smoke set 的检索结果为：

| Controller | 预算分配 | Sentence Recall | 完整句级证据 | Document Recall | 完整文档证据 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单轮 BM25 | `k=8` | 65.69% | 36.00% | 75.75% | 53.00% |
| Scripted two-hop | `k=7 + k=1` | 66.78% | 38.00% | 76.75% | 55.50% |

这只是用于验证环境和查询改写方向的 smoke baseline，不是最终论文级结果。当前 finalizer 主动 abstain，因此 Answer EM/F1 和 Joint Success 均为 0；不能把上述证据召回提升表述成端到端问答提升。smoke 内部 validation 只有 18 条、test 只有 20 条，也不足以支撑显著性结论；主要结果以下面的官方 validation 1,000 条固定 benchmark/dev 为准。

官方 validation 1,000 条固定 benchmark/dev 已完成 BM25 与 scripted two-hop 评测。数据包含
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

# 官方 validation 1k 固定 benchmark/dev；数据需先按 tracked manifest 恢复到对应目录
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

Qwen3-1.7B controller 与同模型 frozen finalizer 已在官方 validation 的固定 100 条诊断子集
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
稳定学习严格动作格式。本轮价值在于发现失败模式，不能写成“SFT 提升”。后续已扩大 teacher
轨迹、构建 SFT-v2 并完成正式 GRPO；这些后续结果单独列在 1.5 节和正式实验矩阵中，不能倒推
本轮小样本 SFT 有效。

### 1.4 扩大版 QLoRA SFT 主结果

官方 train 前 2,000 行中显式审计并跳过 1 条支持句编号越界记录，得到 1,999 个有效任务和
`1617/191/191` 的文档隔离切分。Scripted teacher 在 1,617 个 train episodes 上生成轨迹，
其中 247 个达到 `JointSuccess=1`；严格筛选后形成 741 条 controller decision。Qwen3-1.7B
QLoRA 使用 2 epochs / 94 steps，训练耗时 587.37 秒，train loss 为 0.21865。

扩大版 SFT 与 prompt-only 在同一批官方 validation 1,000 条固定 benchmark/dev 上使用相同 sampling
参数 `temperature=0.7, top_p=0.8, top_k=20`：

| Controller | Answer EM | Answer F1 | Joint Success | 完整句级证据 | 完整文档证据 | 重复调用率 | 预算终止率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only | 18.80% | 25.31% | 11.70% | 41.30% | 61.10% | 63.68% | 100.00% | 4.998 |
| QLoRA SFT-741 | 19.90% | 26.44% | 11.50% | 35.00% | 56.50% | 0.00% | 0.00% | 1.712 |
| SFT - Prompt | +1.10pp | +1.13pp | -0.20pp | -6.30pp | -4.60pp | -63.68pp | -100.00pp | -3.286 |

配对检查确认 1,000 个 task id 完全一致。Answer EM 的反转计数为 41/52，McNemar exact
p=0.300；Joint Success 为 38/36，p=0.908，均不能支持端到端提升。完整句级证据的反转计数
为 98/35（p=4.29e-8），完整文档证据为 94/48（p=1.41e-4），下降具有统计证据。

行为控制改善明确：1000/1000 个 prompt-only episode 因预算耗尽停止，而 SFT 的 1000/1000
个 episode 都主动结束；按 episode 统计，含重复调用的任务从 999 个降到 0 个。平均工具调用
减少 65.7%，每次 Joint Success 对应的工具调用从 42.72 降到 14.89。代价是证据收集不足，
所以本轮只能证明停止策略和调用成本被优化，不能证明问答准确率或 Joint Success 提升。

报告已跟踪：

- `data/agent_rl/reports/qwen3_1.7b_teacher_train2k.json`；
- `data/agent_rl/reports/qwen3_1.7b_sft2k_train.json`；
- `data/agent_rl/reports/qwen3_1.7b_prompt_validation1000.json`；
- `data/agent_rl/reports/qwen3_1.7b_sft2k_validation1000.json`；
- `data/agent_rl/reports/qwen3_1.7b_sft2k_paired_analysis.json`。

### 1.5 SFT-v2 数据修正与 GRPO 训练闭环

针对 SFT-741 过早停止、完整句级证据下降的问题，本轮仍使用同一批官方 HotpotQA train
teacher 轨迹，但把筛选标准改为 `CompleteSentenceEvidence=1`。在 1,617 条轨迹中保留
583 条证据完整轨迹，得到 1,749 条原始决策（工具调用 1,166、停止 583）。只重复每条轨迹
停止前的最后一次工具动作，将训练集重平衡为 2,915 条决策（工具调用 2,332、停止 583，比例
4:1）。问题、答案、证据、messages 和 action 都来自原始官方轨迹，没有新增或改写事实；输入、
输出和报告均保存 SHA-256。

Qwen3-1.7B SFT-v2 使用 2 epochs / 366 steps，训练 2,915 条决策，耗时 2,721.59 秒，
train loss 为 0.12492。随后从 SFT-v2 adapter 初始化正式 GRPO：任务源共 1,617 条，正式运行
限制为 128-task pool；20 updates 实际使用 40 个唯一任务组，每个 update 包含 2 个任务、每个任务
4 条 rollout，共 160 episodes / 525 decisions。训练耗时 1,439.05 秒，峰值 CUDA reserved 约
10.22 GiB。训练只对模型生成的 action token 计算损失，
reference adapter 全程冻结且状态哈希前后一致，最终 policy adapter 已保存并通过独立重载验证。

正式训练报告同时记录官方数据文件、基础模型配置/权重分片、训练代码、配置、初始/最终 adapter
和运行依赖的 SHA-256 或精确版本。上述训练规模足以证明多轮 GRPO 链路真实执行了 rollout、
backward 和参数更新，但不能单凭训练 reward 宣称 benchmark 效果提升；主要结论以下面的同任务
配对评测为准。

固定 1,000 条 benchmark/dev 表明，SFT-v2 将旧 SFT-741 的完整句级/文档级证据从 35.0%/56.5%
恢复到 39.6%/61.2%，但平均工具调用从 1.712 增至 3.690。GRPO 再把调用数降到 2.096、
重复调用率降到 1.23%，同时保持 SFT-v2 的 Joint Success 11.9%；代价是句级/文档级完整证据
降到 37.4%/59.2%。相比 prompt-only，GRPO 的 Answer EM 为 19.9% 对 18.8%、Joint Success
为 11.9% 对 11.7%，对应 paired bootstrap 95% CI 均跨 0，不能宣称端到端显著提升；平均
工具调用减少 58.1%，预算耗尽从 100% 降到 0。当前权威 paired bootstrap 的 Answer EM 95% CI
为 `[-0.8pp, +3.1pp]`，Joint Success 为 `[-1.6pp, +1.8pp]`。句级完整证据下降 3.9pp，
95% CI `[-6.3pp, -1.4pp]`，必须作为方法边界一起报告。所有区间只描述该固定集合和单个
sampling seed，不能替代未参与方案迭代的最终 test。

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

| RL 概念 | 当前已落地 | 后续扩展范围 |
| --- | --- | --- |
| State | 标准化问题、检索观察、证据状态和 step budget，可序列化复现 | 接入长期记忆快照和更多异构工具观察 |
| Action | 严格 `retrieve` / `final_answer` JSON schema，保存 action token 与 log-prob | 扩展 memory、calculator 等动作 |
| Transition | 独立 `reset()` / `step()` 环境接口 | 多环境异步执行 |
| Episode | task ID、seed、环境/策略版本、完整状态和 stop reason | 长期记忆任务的独立 sandbox |
| Reward | task success、evidence coverage、调用成本和错误惩罚的明细 | calculator 与长期记忆安全 reward |
| Trajectory | 原始 messages、action token、old/current/reference log-prob、reward 和 done | 更大规模的列式存储与流式加载 |

训练逻辑已独立于线上 `ToolAgent`，主要模块如下：

```text
app/agent_rl/
├── env.py              # PersonalRAGEnv: reset / step
├── actions.py          # 严格动作 schema 与解析
├── tasks.py            # 任务加载、切分和环境初始化
├── rewards.py          # 可验证 reward 与 reward breakdown
├── policies.py         # prompt-only controller 与严格动作解析
├── finalizers.py       # 冻结、不可见金标答案的终局合成
├── rollouts.py         # policy 决策、transition 与版本化轨迹
├── grpo.py             # 分组 advantage、clipped objective 与 KL
├── comparison.py       # 逐题配对比较与 bootstrap
├── reward_hacking.py   # 正式 reward 防刷审计
├── scripted_agent.py   # 无 GPU 的确定性 controller 与 rollout
├── trajectory.py       # 训练所需 episode/transition schema
├── verifiers.py        # 答案与句级/文档级证据验证
└── adapters.py         # 复用现有 ToolRegistry/RAG/memory 的适配层

scripts/
├── build_agent_sft_data.py
├── train_agent_sft.py
├── train_agent_grpo.py
├── eval_hotpotqa_prompt_policy.py
└── compare_agent_evals.py
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

当前检索主线已使用 1,999 条有效官方 train 任务和 1,000 条官方 validation benchmark/dev。
下面的长期记忆、calculator 和安全任务仍是未来扩展范围，不能用当前 HotpotQA 结果替代：

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

当前实验只使用开源 `hotpotqa/hotpot_qa` distractor 数据；问题、答案、context 与 supporting
facts 均来自官方数据，不自行编造。Scripted teacher 只根据环境 observation 产生检索/停止轨迹，
gold 信息只供 rollout 结束后的 verifier 使用，不进入 policy prompt。官方 train 前 2,000 行
显式跳过 1 条支持句编号越界记录，得到 1,999 个有效任务和 `1617/191/191` 的文档连通分量
隔离切分；官方 validation 前 1,000 条固定为 benchmark/dev，与训练集隔离但已用于诊断和方案
迭代。manifest 记录来源、许可证、坏记录原因和全部文件 SHA-256。

teacher 轨迹通过程序化 verifier 筛选，保留：

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

正式 GRPO reward 以可验证结果为主，以小幅过程奖励辅助学习：

```text
R =
  1.00 * task_success
+ 0.50 * evidence_coverage
- 0.02 * tool_call_count
- 0.20 * invalid_action
- 0.20 * duplicate_call
- 0.50 * premature_final_answer
- 0.30 * budget_exhausted
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

## 7. 已落地修复与剩余边界

### 7.1 Legacy calculator 边界

本轮 HotpotQA Agentic RL 只训练检索与停止动作，正式 finalizer 已消费结构化检索 observation，
不调用 calculator。旧版线上 `ToolAgent._accumulate()` 仍未把 calculator result 交给最终 generator；
因此在未来把 calculator 纳入 RL action space 前，必须先修复并增加端到端测试。当前结果不能外推到
calculator 或长期记忆任务。

### 7.2 Reward hacking 防护状态

正式配置已通过三类机器可读压力测试：

- 完整证据后增加一次无关检索，总回报从 1.48 降为 1.46，coverage 不增加；
- 答案正确但只有 50% 证据时，`task_success=0`，总回报为 -0.27；
- 重复调用、检索失败、伪造参数和伪造引用文本均不能增加 coverage，并受到调用或重复惩罚。

当前审计覆盖检索/证据 reward 主链路，但扩展到 calculator 与长期记忆训练前仍需继续验证：

- 最终答案是否正确；
- 关键结论是否被证据支持；
- 引用是否真实对应证据；
- calculator 输出是否被正确使用；
- 是否存在无意义或重复工具调用；
- 是否发生不必要、错误或敏感的记忆写入。

### 7.3 可训练模型接口状态

线上 `complete_fn` 仍只返回 API 文本，不参与参数更新；训练侧已使用本地开源模型，并在 rollout
时保存 action token IDs、old/current/reference token-level log-prob 和 action mask。

因此：

- OpenAI-compatible API 可继续用于 teacher、数据合成或 judge；
- 被训练的 controller 使用 Qwen3-1.7B + PEFT adapter；
- 当前单卡闭环使用 Transformers + 自定义 clipped GRPO，以兼容现有依赖并保留多轮环境控制；
- 后续需要多 GPU、异步 rollout 或更高吞吐时，再迁移到 verl、vLLM 或 SGLang。

## 8. 必须完成的实验矩阵

当前已完成的主要结果以 1,000 条固定 benchmark/dev 配对实验为准；100 条结果仅保留作诊断：

| 方法 | N | Joint Success | Answer F1 | 完整句级证据 | Invalid Call Rate | Mean Tool Calls | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prompt-only sampling | 1,000 | 11.70% | 25.31% | 41.30% | 0.04% | 5.00 | 正式基线，重复调用和预算耗尽严重 |
| SFT-741 sampling | 1,000 | 11.50% | 26.44% | 35.00% | 0.00% | 1.71 | 行为与成本改善，Joint Success 未提升 |
| Prompt-only sampling（诊断） | 100 | 13.00% | 26.94% | 38.00% | 0.00% | 5.00 | 与 SFT-75 的同批基线 |
| SFT-75 sampling（诊断） | 100 | 4.00% | 16.96% | 11.00% | 12.33% | 1.07 | 小数据导致过早停止，端到端退化 |
| SFT-75 greedy（诊断） | 100 | 4.00% | 18.73% | 16.00% | 13.64% | 0.98 | 退化在 greedy 下仍存在 |
| SFT-v2 sampling（诊断） | 100 | 12.00% | 27.42% | 43.00% | 0.00% | 3.75 | 证据恢复，端到端未超过 prompt-only |
| SFT-v2 + GRPO（诊断） | 100 | 11.00% | 25.99% | 41.00% | 0.00% | 2.08 | 调用和重复继续下降，端到端未提升 |
| SFT-v2 sampling | 1,000 | 11.90% | 25.97% | 39.60% | 0.00% | 3.69 | 证据较 SFT-741 恢复，调用数回升 |
| SFT-v2 + GRPO | 1,000 | 11.90% | 26.16% | 37.40% | 0.00% | 2.10 | Joint 持平，调用减少 43.2%，证据下降 2.2pp |

下一阶段仍需在未参与方案迭代的 final test、多 seed 下补齐 Tool F1、Calls / Success、Unsafe Memory Write Rate。
SFT + GRPO 训练闭环本身已完成；只有新集合对照也支持时，才可形成泛化提升结论。

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

当前可写的真实版本见 `docs/RESUME_AGENTIC_RL_CN.md`。本轮可写“在 1,000 条固定 benchmark/dev 上将
预算耗尽与重复调用降至 0、平均工具调用减少 65.7%”；同时必须说明 Answer EM 差异不显著、
Joint Success 未显著提升。当前可写“完成多轮 GRPO 训练闭环与正式单卡训练，并在固定 1,000 条
benchmark/dev 上将平均工具调用减少 58.1%、预算耗尽降至 0”；不能写“GRPO 显著提高了问答效果”。

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

- 已明确边界：当前 HotpotQA AgentRL 不调用 calculator；legacy ToolAgent 的 calculator 合成仍属未来扩展；
- 已完成：答案与句级/文档级证据 verifier，以及正式 reward 配置的 3/3 防刷压力测试；
- 已完成：按全部可见 context 文档连通分量建立 train/validation/test 切分，并重建 smoke 数据与报告；
- 已完成：准备并评测 1,000 条官方 validation 固定 benchmark/dev，保存内容哈希 manifest 与 BM25/scripted 报告；
- 已完成：AutoDL 上的 Qwen3-1.7B frozen finalizer + prompt-only controller 100 条诊断子集完整指标报告；
- 部分完成：环境状态 reset、seed 和可复现轨迹已落地，memory store 独立快照仍待做。

### P1：完成训练闭环

- 已完成小规模诊断：从 162 个 teacher episodes 筛选 25 个成功 episode，训练 75 条 decision；
- 已完成扩大版：从 1,617 个 teacher episodes 筛选 247 个成功 episode，训练 741 条 decision；
- 已完成：扩大版 SFT 与 prompt-only 在官方 1,000 条固定 benchmark/dev 上的完整配对评测；
- 已确认：停止策略和调用成本明显改善，但 Joint Success 未提升、证据完整度下降；
- 已完成：从 583 条完整句级证据轨迹构建 SFT-v2 的 2,915 条 decision，并完成 366-step QLoRA；
- 已完成：接入 GRPO rollout、action-token loss、冻结 reference、clipped objective、KL、参数更新和 adapter 保存；
- 已完成：20-update 正式 GRPO，128-task capped pool 中实际使用 40 个唯一任务组，160 episodes / 525 decisions；
- 已完成：Prompt-only、SFT-741、SFT-v2、SFT-v2 + GRPO 的官方 1,000 条同任务对照与配对统计。

### P2：做成算法项目

- 完成 reward、数据量、步数预算等消融；
- 分析失败轨迹和训练稳定性；
- 已完成：输出完整技术报告和机器可读结果；
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
