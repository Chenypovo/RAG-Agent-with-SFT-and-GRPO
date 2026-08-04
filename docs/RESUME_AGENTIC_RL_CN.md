# Personal RAG / Agentic RL 简历表述（中文）

> 更新时间：2026-08-04。下面分为“现在可写”和“完成扩大实验后才能写”两版，不能混用。

## 今晚可以写的真实版本

项目名称：Personal RAG Agent 训练与评测环境

- 构建可复现的多轮 RAG Agent 环境，将检索、停止决策和最终回答拆成严格 JSON action、环境 transition 与分项 reward；实现输入哈希校验、完整轨迹落盘及 Answer、证据覆盖、Joint Success、非法/重复调用等端到端指标。
- 基于 Qwen3-1.7B 在 HotpotQA 官方 validation 独立测试集完成 100 题 prompt-only 基线：Answer EM 21.0%、Answer F1 26.94%、Joint Success 13.0%、完整句级证据覆盖 38.0%；同时定位重复调用率 63.0%、预算终止率 100% 的策略缺陷。
- 从 162 条训练任务中筛选 25 条 Joint Success 轨迹、生成 75 条 controller 决策样本，完成 QLoRA SFT 小样本试验（2 epochs、10 steps、60.37 秒、loss 1.261）；held-out 结果显示模型出现过早停止，Joint Success 从 13.0% 降至 4.0%，据此将扩大轨迹规模与失败恢复数据列为后续重点。

### 面试时必须主动说明

- 当前 SFT 是 75 条 decision 的小样本诊断实验，端到端效果退化，不能写成“性能提升”。
- Prompt-only 与 SFT 使用同一 Qwen3-1.7B base model、同一 frozen finalizer、同一批 100 条 held-out 输入和相同 seed；sampling 对照的 controller 参数均为 `temperature=0.7, top_p=0.8, top_k=20`。
- 当前尚未完成 GRPO 训练、SFT + GRPO 主结果和消融实验，简历中不得出现“通过 GRPO 提升了 X%”。
- 100 条 held-out 只能作为阶段结果；正式结论仍需扩大训练数据，并在完整 held-out 或多 seed 上复验。

## 扩大数据并验证后才能升级的版本

以下是模板，不是当前成果。只有当占位数字都由可复现实验报告支持后才能使用：

- 构建可复现的 Personal-RAG Agent 后训练环境，在按全部可见 context 文档隔离的数据切分上生成并筛选 **X 条任务 / Y 条成功轨迹 / Z 条决策样本**，完成 Qwen3-1.7B QLoRA SFT 与 GRPO 多轮工具调用训练。
- 在 **N 条 held-out 任务、M 个随机种子**上，将 Joint Success 从 prompt-only 的 **A%** 提升至 **B%**，Answer F1 从 **C%** 提升至 **D%**，同时将非法动作率从 **E%** 降至 **F%**、每次成功平均工具调用数从 **G** 降至 **H**。
- 完成 outcome-only/shaped reward、工具成本、步数预算和数据规模消融，并通过失败轨迹分析验证收益不是来自提前停止、重复调用或 verifier 漏洞。

升级前最低条件：

1. 扩大 teacher 成功轨迹，覆盖正常结束、失败恢复和不同调用长度，而不是只复制少量成功模板；
2. SFT 在同一 held-out、多 seed 下至少不低于 prompt-only，再进入 GRPO；
3. 跑通 SFT + GRPO，并保存训练曲线、reward 分量、KL/entropy、stop reason 与完整评测报告；
4. 完成至少三组消融和人工失败案例审查；
5. 所有简历数字都能对应到仓库报告与输入哈希。
