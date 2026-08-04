# Personal RAG + 长期记忆 Agent

轻量级个人 RAG + 长期记忆 Agent：多模态文档检索增强问答，跨会话沉淀长期记忆，由路由层每轮自主决定「查文档 / 调记忆 / 写记忆」。

**技术栈**：Python · LanceDB/FAISS · BM25 · BGE Reranker · OpenAI-compatible / 本地 HF embedding · FastAPI

## 功能

- **多模态接入**：`txt / md / pdf / image(OCR) / video`
- **结构感知切块**：词元感知（tiktoken）+ 段落/句子边界；chunk 保留**标题层级**、**页码**、**块类型**（heading/list/table/code/paragraph）、切分边界、多模态路径等 metadata
- **向量库**：LanceDB 统一行存（id/vector/text/metadata），与 FAISS（IndexFlatL2）两套后端结果等价；Agent 默认走 LanceDB
- **混合检索**：向量 + BM25 经 RRF 融合 → BGE Reranker 精排
- **parent-child 召回**：索引小子块保精度，命中后按**标题章节**扩展回父块（无标题则相邻窗口），父块去重
- **受控生成**：证据约束回答、按 source/chunk 展示依据，证据不足时拒答
- **长期记忆**：从对话抽取事实三元组 → LLM 判定 add/update/delete 归并 → SQLite（事实）+ 独立向量索引（embedding）持久化；提问时按 embedding cosine 召回注入。**防误判（规则化判定）**：UPDATE 需**同槽位**（fact_object 一致）+ cosine ≥ 0.8 → 非销毁 supersede（旧置 SUPERSEDED 可恢复），否则降级 ADD；DELETE 更危险——需**矛盾证据**（带 content）+ 同槽位 + 更高门槛 0.9，否则保留旧事实。把 add/update 做成确定性判定，只把"矛盾型删除"留给 LLM
- **对话 Agent**：路由决策 + 记忆增强生成（单次基线）；**工具调用循环 ToolAgent**——自研控制循环 + 工具注册表 + 失败重试/预算/死循环护栏 + 轨迹日志，全程 prompt-based JSON、可离线单测（见下文）
- **Web**：FastAPI 后端 + 自研前端（`webui/`），另提供 Streamlit
- **评测**：检索 & 记忆的 `Recall@K` / `MRR@K`

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 LLM key；embedding 可选远端或本地
```

`.env` 关键项（embedding 二选一）：

```bash
LLM_PROVIDER=openai_compatible      # 回答/抽取/路由走 API
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_BASE_URL=...
EMBED_PROVIDER=local                # 本地：BAAI/bge-small-zh-v1.5（或 openai_compatible）
EMBED_MODEL=BAAI/bge-small-zh-v1.5
```

建库 → 跑应用：

```bash
# 1) 把文档放进 data/uploads，建索引（向量 + BM25）
python scripts/build_index.py --input-dir data/uploads --embed-backend local

# 2) Web 端（聊天 + 实时展示沉淀的记忆）
uvicorn app.api.server:app --port 8000      # 浏览器打开 http://127.0.0.1:8000

# 或命令行多轮对话
python scripts/chat_demo.py
```

检索单测：`python scripts/query_demo.py --retrieval-backend hybrid --rerank --query "..." --show-chunks`
（`--vector-store faiss|lancedb`、`--embed-backend local|openai|clip` 等参数见 `--help`）

## 评测

```bash
# 检索（chunk 级 qrels，标到含答案的 chunk；加 --rerank 走完整链路）
python scripts/eval_retrieval.py --vector-store lancedb --lancedb-uri data/index_eval/lancedb \
  --bm25-path data/index_eval/bm25.json --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl --backend all --ks 1,4 --embed-backend local --no-strict [--rerank]

# 记忆召回
python scripts/eval_memory.py --facts data/eval/memory_facts_synth.jsonl \
  --queries data/eval/memory_queries_synth.jsonl --qrels data/eval/memory_qrels_synth.jsonl --ks 1,3
```

实验结果（合成语料 40 查询，本地 bge-small，LanceDB；FAISS 数字逐位相同）。
左两列＝无精排基线，右两列＝加 `--rerank` 完整链路：

| Backend | 基线 Recall@1 | 基线 MRR@4 | +精排 Recall@1 | +精排 MRR@4 |
| --- | ---: | ---: | ---: | ---: |
| 向量 | 0.5125 | 0.6896 | 0.7625 | 0.8792 |
| BM25 | 0.7375 | 0.8521 | 0.7875 | 0.9042 |
| **混合** | 0.6000 | 0.8042 | **0.7875** | **0.9042** |

- **完整链路（混合 + BGE 精排）：Recall@1=0.79 · Recall@4=0.975 · MRR@4=0.90**；混合无精排时 Recall@4=1.00。
- **`eval_retrieval.py` 默认不精排；加 `--rerank` 才走 BGE 精排。**
- **qrels 标到 chunk 级**（含答案的 chunk）。若只标到文档级（脚本会塌成 `#0`），会把"召回到正确文档的其它 chunk"误判为错，**低估** MRR/Recall@1（文档级会让混合 MRR@4 从 0.90 假性掉到 0.66）。

### parent-child 父块窗口消融（`scripts/eval_parent_window.py`）

heading-less 细粒度语料（104 chunk），命中后扩展到 W 个连续子块的父块，测**答案覆盖率**与**上下文 token**：

| W | 覆盖率(top-4) | 上下文 tokens | 说明 |
| ---: | ---: | ---: | --- |
| 1（child-only） | 0.900 | 349 | 基线 |
| 2 | 0.900 | 612 | 零增益、翻倍 token |
| **3** | **0.975** | 891 | 覆盖率拐点 |
| 5 | 0.975 | 1143 | 不再涨、纯增成本 |

结论：**W=3 是覆盖率/成本的拐点**（W=2 无增益、W=5 仅增 token），故无标题文档默认窗口取 3；有标题文档则按章节边界分组。

父块大小统一**封顶**（`max_parent_chunks`，默认 6，居中命中）：窗口模式天然 ≤W，章节模式也不会因某节过大而灌爆 LLM 上下文。
- 完整链路（混合 + BGE 精排）：**Recall@1=0.50 · Recall@4=0.89 · MRR@4=0.70**；精排把 top-1 命中从 0.40 提到 0.50。
- 记忆召回（`memory_synth_report.json`，31 事实 / 20 查询带干扰，**与 reranker 无关**）：**Recall@1=0.90 · Recall@3=0.95 · MRR@1=0.90**

> 评测集为自建标注集（`data/eval/`）。可把建库后的 `metadatas.json` + 原文喂给 LLM 自动生成 `queries/qrels`。

## 工具调用 Agent 循环（ToolAgent）

把「一次 LLM 调用出三个开关」的固定路由（`Router`，保留作基线）升级为真正的工具调用 agent：

```
用户消息 → ToolAgent 循环（app/agent/loop.py，≤ max_steps）
  每步: 渲染 prompt(工具清单+plan+已执行步骤观察) → LLM 出一个 JSON 动作
        ├─ {"tool", "args"}      → ToolRegistry 校验+分发（retrieve_docs / read_memory / write_memory / calculator）
        ├─ {"plan": [...]}       → 覆盖待办计划（可与 tool 同步出现）
        └─ {"final_answer":true} → 循环特判收尾（不走注册表）→ 复用现成 generate_fn
                                    对累积证据做受控/带引用/证据不足拒答的终局合成
  工具失败/证据不足 → 错误进观察，下一轮模型自行纠正（反思重试）
```

- **全程 prompt-based JSON**：不依赖 provider 原生 function calling，`complete_fn` 注入即可换模型；防御式解析 + 重试兜底。
- **多跳=多次 retrieve_docs**：拆解体现在 agent 用不同子查询连续检索，证据跨调用累积去重后统一合成。
- **护栏全覆盖且有测试**：非法 JSON（连续超限 `parse_abort`）、未知工具/坏参数（注册表侧校验回喂）、工具内部异常兜成 `ok=False`、步数/调用双预算（`budget` 用已收集证据兜底合成）、完全相同调用的死循环检测（提示注入，连续重复超限 `loop_abort`）。任何终止路径都返回合法结果，不向上抛异常。
- **轨迹可观测**：每步 `{thought, tool, args, observation, ok, error, latency_ms}` 随结果返回，`format_trajectory()` 可打印。
- 装配：`from app.agent.factory import build_tool_agent`（与 `build_memory_agent` 共享同一套检索/记忆/生成运行时）。

### 评测（基线 vs 循环，`scripts/eval_agent.py`）

```bash
# 生成多步复合任务（多跳/记忆依赖/计算三类，生成后需人工抽查）
python scripts/gen_agent_tasks.py --n 24 --out data/eval/agent_tasks_synth.jsonl

# 同一批任务上对照：单次基线（MemoryAgent）vs 工具循环（ToolAgent）
python scripts/eval_agent.py --tasks data/eval/agent_tasks_synth.jsonl
```

程序化指标：任务成功率（多跳看 gold chunk 全覆盖、计算看数值命中、记忆看关键词）、多跳 Coverage@k、工具选择 precision/recall、成本（每任务 LLM 调用/工具调用/步数/解析失败率）。报告落盘 `data/eval/agent_eval_report.json`。

| 指标 | 基线（单次路由） | ToolAgent 循环 |
| --- | ---: | ---: |
| 任务成功率 | 待跑 | 待跑 |
| 多跳 Coverage@8 | 待跑 | 待跑 |
| 工具选择 P / R | 待跑 | 待跑 |
| LLM 调用/任务 | 待跑 | 待跑 |
| 解析失败率 | — | 待跑 |

> 表内数字待 LLM API 恢复后运行上述两条命令回填（评测链路已跑通，当前卡在 API 余额）。示例任务集 `data/eval/agent_tasks_example.jsonl` 已入库可直接冒烟。

**诚实边界**：
1. prompt-based JSON 工具调用**不如**原生 function calling 稳，靠防御解析+重试兜底，偶发解析失败用**解析失败率**量化，不藏。
2. 评测集小、LLM 合成，只信**相对结论**（与本项目其它评测口径一致）。
3. 多步天然比单次**更慢更贵**，对照表直接报出每任务多出的调用次数，不美化。
4. `calculator` 是"异构工具"的离线演示，体量有限；联网工具留 Phase 2。
5. 记忆工具严格复用现有非破坏归并（同槽位+相似度门控），本期未做记忆侧新消融。

## Agentic RL 训练与评测环境

项目已增加独立的 `app/agent_rl/`，用于把检索 Agent 表达成可训练、可验证的多轮环境。目前已完成
prompt-only 基线与一次 75 条 decision 的 QLoRA SFT 小样本诊断实验；SFT 端到端效果退化，
GRPO 尚未完成，不能把“训练跑通”写成“性能提升”：

- `reset()/step()` 环境、严格 JSON action、step budget、重复调用惩罚和分项 reward；
- HotpotQA 转换、按全部可见 context 文档连通分量隔离的数据切分、BM25 与确定性双轮检索 baseline；
- 冻结答案生成器和 prompt-only controller 评测；
- 保存原始模型输出、prompt、解析错误、transition、seed、环境/策略版本和端到端 verifier 指标。

```bash
# 准备 verifier-ready HotpotQA 数据与 BM25 索引
python scripts/prepare_hotpotqa_agent_rl.py --limit 200 \
  --output-dir data/agent_rl/hotpotqa_smoke

# Dataset Viewer 限流时，可从官方 Parquet 分片离线读取指定窗口
# `gold_document_hash` 是兼容旧命令的参数名，实际按全部可见 context 文档连通分量切分
python scripts/prepare_hotpotqa_agent_rl.py --limit 2000 \
  --input-parquet /path/to/train-shard.parquet \
  --source-split train --partition-mode gold_document_hash \
  --output-dir data/agent_rl/hotpotqa_train_2k

# 无模型的检索方向 smoke baseline
python scripts/eval_hotpotqa_scripted_agent.py \
  --data-dir data/agent_rl/hotpotqa_smoke --partition all

# 冻结模型的端到端 prompt-only baseline
python scripts/eval_hotpotqa_prompt_policy.py \
  --data-dir data/agent_rl/hotpotqa_smoke --partition test \
  --env-file /path/to/.env

# 或在 GPU 机器直接用同一个开源模型完成 controller + frozen finalizer
python scripts/eval_hotpotqa_prompt_policy.py \
  --completion-backend transformers \
  --controller-model Qwen/Qwen3-1.7B \
  --finalizer-model Qwen/Qwen3-1.7B \
  --data-dir data/agent_rl/hotpotqa_smoke --partition test
```

数据本体不入库；内容哈希、切分规模和无模型基线报告保存在
`data/agent_rl/manifests/` 与 `data/agent_rl/reports/`。
整夜扩展训练使用官方 train 前 2,000 行：显式审计并跳过 1 条支持句编号越界的坏记录，
得到 1,999 个有效任务和 `1617/191/191` 的 train/validation/test 切分；对应清单为
`data/agent_rl/manifests/hotpotqa_train_2k.json`。

已在官方 HotpotQA validation 的前 1,000 条样本上完成独立 test 评测。为公平比较，
单轮 BM25 与 scripted two-hop controller 都只取 8 条检索结果：

| Controller | 预算 | Sentence Recall | 完整句级证据 | Document Recall | 完整文档证据 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单轮 BM25 | `k=8` | 64.62% | 34.70% | 75.80% | 55.00% |
| Scripted two-hop | `k=7 + k=1` | 64.92% | 35.70% | 75.65% | 55.10% |
| 差值 | — | +0.31 个百分点 | +1.00 个百分点 | -0.15 个百分点 | +0.10 个百分点 |

这组增益很小且方向不完全一致，只能说明两跳查询改写链路可运行，不能说明它稳定优于单轮
BM25。scripted 评测没有答案生成器，会主动拒答，因此 Answer EM/F1 与 Joint 指标均为 0。
完整报告见 `data/agent_rl/reports/hotpotqa_validation_1k_*.json`。

Qwen3-1.7B controller 与同模型 frozen finalizer 已在官方 validation 的固定 100 条 held-out
任务上完成端到端对照。Sampling 均使用 `temperature=0.7, top_p=0.8, top_k=20`；greedy 为
`temperature=0`：

| Controller | Answer EM | Answer F1 | Joint Success | 完整句级证据 | 重复调用率 | 预算终止率 | 非法动作率 | 平均工具调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-only，sampling | 21.00% | 26.94% | 13.00% | 38.00% | 63.00% | 100.00% | 0.00% | 5.00 |
| QLoRA SFT-75，sampling | 12.00% | 16.96% | 4.00% | 11.00% | 1.32% | 0.00% | 12.33% | 1.07 |
| QLoRA SFT-75，greedy | 13.00% | 18.73% | 4.00% | 16.00% | 1.36% | 5.00% | 13.64% | 0.98 |

训练数据来自 162 个 teacher episodes 中通过 `JointSuccess=1` 筛选出的 25 个成功 episode，
共 75 条 controller decision。QLoRA 训练 2 epochs / 10 steps，用时 60.37 秒，最终 loss 1.261。
小数据 SFT 虽然减少了重复调用和预算耗尽，却明显过早停止，并引入 12%～14% 非法动作，导致
Answer 与 Joint Success 退化；因此本轮是失败诊断，不是 SFT 提升结论。

原始可复现报告位于 `data/agent_rl/reports/qwen3_1.7b_*validation100*.json`。今晚可用于简历的

完整进度、实验边界和 SFT/GRPO 计划见

## 结构

`app/`：`loader`（多模态+OCR）· `chunker`（结构感知切块）· `embedder` · `vectordb`（LanceDB/FAISS/BM25）· `retriever`（含 parent-child）· `reranker` · `generator` · `memory`（抽取/归并/召回/存储）· `agent`（路由基线 + 工具循环：`loop`/`registry`/`tools`/`trajectory`）· `api`（FastAPI）· `eval`
