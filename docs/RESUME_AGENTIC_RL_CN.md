# Personal RAG / Agentic RL 简历表述（中文）

> 更新时间：2026-08-06。以下数字均来自可复现报告；不包含尚未完成的多 seed 或消融结论。

## 推荐直接粘贴版

项目名称：可验证多轮 RAG Agent 后训练系统

- 构建可复现的多轮 RAG Agent 环境，将检索与停止决策建模为严格 JSON action、环境 transition 和分项 reward；实现 action-token 轨迹、输入/模型/代码哈希，以及 Answer、证据覆盖、Joint Success、非法/重复调用等端到端评测。
- 基于开源 HotpotQA 构建文档连通分量隔离的数据链路，从 1,617 条官方 teacher 轨迹筛选 583 条完整句级证据轨迹，重平衡为 2,915 条 controller 决策（工具/停止=2,332/583）；训练内容逐行回溯原始 rollout，不合成问题、答案或证据。
- 基于 Qwen3-1.7B 完成 QLoRA SFT 与 20-update 多轮 GRPO；从 1,617 条训练任务中限制 128-task pool，实际使用 40 个唯一任务组，生成 160 个 rollout episodes / 525 条决策；实现同任务分组 advantage、action-token clipped objective、冻结 reference KL、数值稳定性检查、adapter 保存重载和全链路 provenance。
- 在官方 validation 1,000 条固定 benchmark/dev 的逐题配对评测中，相比 prompt-only 将平均工具调用从 4.998 降至 2.096（-58.1%）、重复调用率从 63.68% 降至 1.23%、预算耗尽率从 100% 降至 0；Joint Success 为 11.9% 对 11.7%，Answer EM 为 19.9% 对 18.8%。
- 对正式 reward 配置执行防刷审计与 paired bootstrap：无关、重复、失败和伪造证据行为不能增加 coverage；Answer EM 与 Joint Success 差值的 95% CI 均跨 0，不宣称准确率显著提升，并明确报告完整句级证据下降 3.9pp 的代价。

## 更短的三条版本

- 构建可验证多轮 RAG Agent 后训练环境，基于官方 HotpotQA 完成文档隔离的数据构造、严格 action/trajectory、分项 reward、防刷审计及逐题配对评测。
- 基于 Qwen3-1.7B 完成 2,915 条决策的 QLoRA SFT 与 20-update GRPO，实现同任务分组 rollout、action-token clipped objective、冻结 reference KL、adapter 保存重载和全链路哈希追踪。
- 在 1,000 条官方固定 benchmark/dev 上将平均工具调用减少 58.1%、重复调用率从 63.68% 降至 1.23%、预算耗尽率降至 0，同时维持 Joint Success 11.9% 对 11.7%；主动披露准确率差异不显著及证据完整度下降边界。

## 面试时必须主动说明

- Prompt-only、SFT-v2 和 GRPO 使用同一 Qwen3-1.7B base model、同一 frozen finalizer、同一批 1,000 条输入和相同 seed；sampling 参数均为 `temperature=0.7, top_p=0.8, top_k=20`。这批数据与训练集隔离，但曾用于失败诊断和方案迭代，不是 untouched final test。
- 这是单个 sampling seed。工具成本和预算终止的差异很大且配对区间不跨 0，但不能推广为多 seed 稳定结论。
- GRPO 相比 prompt-only 的 Answer EM 为 +1.1pp，95% CI `[-0.8pp, +3.1pp]`；Joint Success 为 +0.2pp，95% CI `[-1.6pp, +1.8pp]`，不能写成准确率显著提升。
- 完整句级证据从 41.3% 降至 37.4%，差值 -3.9pp，95% CI `[-6.3pp, -1.4pp]`。项目结论是行为效率优化，不是全面质量提升。
- SFT-v2 相比旧 SFT-741 将完整句级/文档级证据从 35.0%/56.5% 恢复到 39.6%/61.2%，但平均工具调用从 1.712 回升到 3.690；GRPO 再将调用降到 2.096，同时牺牲 2.2pp 句级完整证据。

## 下一阶段才能增加的表述

只有完成以下工作后，才可增加“稳定提升端到端效果”或“完成系统消融”的描述：

1. 多个 sampling seed 复验；
2. outcome-only、shaped reward 和无工具成本三组消融；
3. max steps、group size 和训练数据规模消融；
4. 补充困难样本与失败恢复轨迹，使证据完整度恢复到不低于 prompt-only；
5. 人工审查错误案例和开放式 reward 漏洞。
