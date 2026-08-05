# Personal RAG / Agentic RL 简历表述（中文）

> 更新时间：2026-08-05。下面分为“现在可写”和“完成 GRPO 后才能写”两版，不能混用。

## 现在可以写的真实版本

项目名称：Personal RAG Agent 训练与评测环境

- 构建可复现的多轮 RAG Agent 环境，将检索、停止决策和最终回答拆成严格 JSON action、环境 transition 与分项 reward；实现输入哈希校验、完整轨迹落盘及 Answer、证据覆盖、Joint Success、非法/重复调用等端到端指标。
- 基于 HotpotQA 构建文档隔离的训练/验证切分；在 1,617 条 teacher 轨迹中筛选 247 条 Joint Success 轨迹，生成 741 条 controller 决策样本，完成 Qwen3-1.7B QLoRA SFT（2 epochs、94 steps、587 秒）。
- 在官方 validation 1,000 条独立 held-out 的同任务配对评测中，将预算耗尽率从 100% 降至 0、重复调用率从 63.68% 降至 0、平均工具调用从 4.998 降至 1.712（减少 65.7%）；Answer EM 为 19.9% 对 18.8%，差异不显著（p=0.300），Joint Success 为 11.5% 对 11.7%。
- 对逐题轨迹执行配对检验与失败分析，定位扩大版 SFT 的主要边界：停止和调用成本得到控制，但完整句级证据从 41.3% 降至 35.0%，下一阶段需补充长轨迹、困难样本和失败恢复训练。

### 面试时必须主动说明

- 当前扩大版 SFT 是 741 条 decision、单个 sampling seed 的实验；它改善的是停止策略与工具成本，不是端到端成功率。
- Prompt-only 与 SFT 使用同一 Qwen3-1.7B base model、同一 frozen finalizer、同一批 1,000 条 held-out 输入和相同 seed；sampling 对照参数均为 `temperature=0.7, top_p=0.8, top_k=20`。
- 当前尚未完成 GRPO 训练、SFT + GRPO 主结果和消融实验，简历中不得出现“通过 GRPO 提升了 X%”。
- Answer EM 的 +1.1 个百分点不显著，Joint Success 还下降 0.2 个百分点；不得写成“准确率提升”或“SFT 全面优于基线”。

## 完成 GRPO 并验证后才能升级的版本

以下是模板，不是当前成果。只有当占位数字都由可复现实验报告支持后才能使用：

- 构建可复现的 Personal-RAG Agent 后训练环境，在按全部可见 context 文档隔离的数据切分上生成并筛选 **X 条任务 / Y 条成功轨迹 / Z 条决策样本**，完成 Qwen3-1.7B QLoRA SFT 与 GRPO 多轮工具调用训练。
- 在 **N 条 held-out 任务、M 个随机种子**上，将 Joint Success 从 prompt-only 的 **A%** 提升至 **B%**，Answer F1 从 **C%** 提升至 **D%**，同时将非法动作率从 **E%** 降至 **F%**、每次成功平均工具调用数从 **G** 降至 **H**。
- 完成 outcome-only/shaped reward、工具成本、步数预算和数据规模消融，并通过失败轨迹分析验证收益不是来自提前停止、重复调用或 verifier 漏洞。

升级前最低条件：

1. 补充长轨迹、困难样本和失败恢复数据，避免 SFT 只学习到更早停止；
2. 在多 seed 下复验行为收益，并把完整证据率恢复到不低于 prompt-only；
3. 跑通 SFT + GRPO，并保存训练曲线、reward 分量、KL/entropy、stop reason 与完整评测报告；
4. 完成至少三组消融和人工失败案例审查；
5. 所有简历数字都能对应到仓库报告与输入哈希。
