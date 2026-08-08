# Personal RAG + 长期记忆 Agent

[![tests](https://github.com/Chenypovo/Personal_RAG/actions/workflows/tests.yml/badge.svg)](https://github.com/Chenypovo/Personal_RAG/actions/workflows/tests.yml)

面向私有知识库与工作区任务的可扩展多步工具 Agent：统一编排文档检索、长期记忆、计算和只读文件工具；保留单步固定路由作为对照基线。

**技术栈**：Python · LanceDB/FAISS · BM25 · BGE Reranker · OpenAI-compatible / 本地 HF embedding · FastAPI

## 功能

- **数据加载**：支持 `txt / md / pdf / image(OCR) / video`；当前主检索评测只覆盖文本语料，图片 OCR 与视频帧尚未纳入主结果
- **边界感知切块与元数据**：按 token 预算并优先在段落/句子边界截断；chunk 保留标题层级、页码、块类型、字符边界和多模态路径等 metadata
- **向量库**：LanceDB 统一行存（id/vector/text/metadata），与 FAISS（IndexFlatL2）两套后端结果等价；Agent 默认走 LanceDB
- **混合检索**：向量 + BM25 经 RRF 融合 → BGE Reranker 精排
- **parent-child 召回**：索引小子块保精度，命中后按**标题章节**扩展回父块（无标题则相邻窗口），父块去重
- **证据分区生成**：文档证据、只读工具结果、用户记忆和已确认写操作分别进入独立上下文；文档结论按 source/chunk 展示依据，信息不足时明确说明
- **长期记忆**：从对话抽取事实三元组 → LLM 判定 add/update/delete 归并 → SQLite（事实）+ 独立向量索引（embedding）持久化；提问时按 embedding cosine 召回注入。**防误判（规则化判定）**：UPDATE 需**同槽位**（fact_object 一致）+ cosine ≥ 0.8 → 非销毁 supersede（旧置 SUPERSEDED 可恢复），否则降级 ADD；DELETE 更危险——需**矛盾证据**（带 content）+ 同槽位 + 更高门槛 0.9，否则保留旧事实。把 add/update 做成确定性判定，只把"矛盾型删除"留给 LLM
- **对话 Agent**：单步固定路由保留为基线；**ToolAgent** 通过统一注册表多步调用文档、记忆、计算及 `glob / grep / read_file` 只读工作区工具，具备失败重试、预算、死循环护栏和轨迹日志
- **Web**：FastAPI 后端 + 自研前端（`webui/`），另提供 Streamlit
- **评测**：检索 & 记忆的 `Recall@K` / `MRR@K`

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

> **部署边界**：当前 Web/API 是本地单用户演示，所有请求共享同一份 `data/memory/`。它尚未实现鉴权、用户/会话隔离和限流，**不要直接暴露到公网或用于多用户服务**。

检索单测：`python scripts/query_demo.py --retrieval-backend hybrid --rerank --query "..." --show-chunks`
（`--vector-store faiss|lancedb`、`--embed-backend local|openai|clip` 等参数见 `--help`）

## 评测

仓库内置一套可公开、冻结的小型检索评测集，新克隆仓库后可直接运行：

- 语料：`data/uploads_eval/` 下 11 篇 Markdown，共 20 个已索引 chunk
- 查询：40 条；标注：46 条二元相关性标注，粒度为 `source#chunk_id`
- 冻结索引：FAISS + BM25，位于 `data/index_eval/`
- 冻结报告：`retrieval_synth_chunk_report.json`（无精排）与 `retrieval_synth_chunk_rerank_report.json`（加精排）

```bash
# 无精排基线
EMBED_MODEL=BAAI/bge-small-zh-v1.5 python scripts/eval_retrieval.py \
  --vector-store faiss \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json \
  --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl \
  --backend all --ks 1,4 --embed-backend local --no-strict \
  --output-json data/eval/retrieval_synth_chunk_report.json

# 完整链路：同一候选集加 BGE reranker
EMBED_MODEL=BAAI/bge-small-zh-v1.5 python scripts/eval_retrieval.py \
  --vector-store faiss \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json \
  --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl \
  --backend all --ks 1,4 --embed-backend local --no-strict --rerank \
  --output-json data/eval/retrieval_synth_chunk_rerank_report.json

# 记忆召回
python scripts/eval_memory.py --facts data/eval/memory_facts_synth.jsonl \
  --queries data/eval/memory_queries_synth.jsonl --qrels data/eval/memory_qrels_synth.jsonl --ks 1,3
```

首次执行会从 Hugging Face 下载本地 embedding/reranker 模型。以下数字只对应上述冻结数据、chunk-level qrels 和本地 `BAAI/bge-small-zh-v1.5`；不把旧的文档级 qrels 结果混入主结论。

| Pipeline | Recall@1 | Recall@4 | MRR@4 |
| --- | ---: | ---: | ---: |
| 向量（无精排） | 0.5125 | 0.8375 | 0.6896 |
| BM25（无精排） | 0.7375 | 0.9000 | 0.8521 |
| 混合（无精排） | 0.6000 | **1.0000** | 0.8042 |
| 向量 + 精排 | 0.7625 | 0.9500 | 0.8792 |
| BM25 + 精排 | **0.7875** | **0.9750** | **0.9042** |
| **混合 + 精排** | **0.7875** | **0.9750** | **0.9042** |

- **完整链路（混合 + BGE 精排）：Recall@1=0.7875 · Recall@4=0.9750 · MRR@4=0.9042**；混合无精排时 Recall@4=1.0000。
- **`eval_retrieval.py` 默认不精排；加 `--rerank` 才走 BGE 精排。**
- **qrels 标到 chunk 级**（含答案的 chunk）。旧文档级标注会把同一文档的其它 chunk 误判为错，因此不再用于主结果。

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
- 记忆召回（`memory_synth_report.json`，31 事实 / 20 查询带干扰，**与 reranker 无关**）：**Recall@1=0.90 · Recall@3=0.95 · MRR@1=0.90**

> 检索评测集为自建冻结集（`data/eval/`）；具体文件、重建方式和口径见 `data/eval/README_synth.md`。

## 工具调用 Agent 循环（ToolAgent）

把「一次 LLM 调用出三个开关」的固定路由（`Router`，保留作基线）升级为真正的工具调用 agent：

```
用户消息 → ToolAgent 循环（app/agent/loop.py，≤ max_steps）
  每步: 渲染 prompt(工具清单+plan+已执行步骤观察) → LLM 出一个 JSON 动作
        ├─ {"tool", "args"}      → ToolRegistry 校验+分发（知识、记忆、计算、只读工作区工具）
        ├─ {"plan": [...]}       → 覆盖待办计划（可与 tool 同步出现）
        └─ {"final_answer":true} → 循环特判收尾（不走注册表）→ 复用现成 generate_fn
                                    对文档证据、工具结果和已确认操作分区后做终局合成
  工具失败/证据不足 → 错误进观察，下一轮模型自行纠正（反思重试）
```

- **全程 prompt-based JSON**：不依赖 provider 原生 function calling，`complete_fn` 注入即可换模型；防御式解析 + 重试兜底。
- **多跳=多次 retrieve_docs**：拆解体现在 agent 用不同子查询连续检索，证据跨调用累积去重后统一合成。
- **护栏全覆盖且有测试**：非法 JSON（连续超限 `parse_abort`）、未知工具/坏参数（注册表侧校验回喂）、工具内部异常兜成 `ok=False`、步数/调用双预算（`budget` 用已收集证据兜底合成）、完全相同调用的死循环检测（提示注入，连续重复超限 `loop_abort`）。任何终止路径都返回合法结果，不向上抛异常。
- **轨迹可观测**：每步 `{thought, tool, args, observation, ok, error, latency_ms}` 随结果返回，`format_trajectory()` 可打印。
- **统一工具输出**：只读结果进入 `ToolArtifact`；写记忆等副作用进入独立 `ConfirmedAction`，可确认“操作已完成”，但不会被当作文档事实证据。
- **安全工作区工具**：`glob_files / grep_text / read_file` 默认关闭，只有显式传入 `workspace_root` 才启用；拒绝绝对路径、父目录穿越和越界软链接，并限制扫描数量、文件大小和返回长度。
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

## 结构

`app/`：`loader`（多模态+OCR）· `chunker`（结构感知切块）· `embedder` · `vectordb`（LanceDB/FAISS/BM25）· `retriever`（含 parent-child）· `reranker` · `generator` · `memory`（抽取/归并/召回/存储）· `agent`（路由基线 + 工具循环：`loop`/`registry`/`tools`/`trajectory`）· `api`（FastAPI）· `eval`
