# Frozen Synthetic Retrieval Eval

本目录包含一套可公开、冻结的小型检索评测集，用于复现 Personal RAG 的 `Recall@K` 和 `MRR@K`。它适合做功能回归和方案对比，不代表生产流量效果。

## 数据规模与口径

- 文档库：`data/uploads_eval/`，11 篇中文 Markdown
- 冻结索引：20 个 chunk
- 查询：`queries_synth.jsonl`，40 条
- 标注：`qrels_synth_chunk.jsonl`，46 条二元相关性标注
- 标注粒度：`source#chunk_id`
- 部分查询对应多个相关 chunk，因此 Recall@K 按相关 chunk 覆盖比例计算

文档和问题覆盖结构化 chunking、LanceDB/FAISS、BM25/RRF、BGE Reranker、长期记忆、证据约束生成、多模态接入和部署配置。

## 已冻结文件

- `data/uploads_eval/*.md`：评测语料
- `data/eval/queries_synth.jsonl`：查询
- `data/eval/qrels_synth_chunk.jsonl`：chunk-level qrels
- `data/index_eval/faiss.index`：FAISS 索引
- `data/index_eval/metadatas.json`：索引 metadata
- `data/index_eval/bm25.json`：BM25 索引
- `data/eval/retrieval_synth_chunk_report.json`：无精排报告
- `data/eval/retrieval_synth_chunk_rerank_report.json`：BGE 精排报告

`queries_synth.jsonl` 每行格式：

```json
{"query_id":"q001","query":"自然语言问题","query_image":""}
```

`qrels_synth_chunk.jsonl` 每行格式：

```json
{"query_id":"q001","doc_id":"data/uploads_eval/rag_architecture.md#0","relevance":1}
```

## 直接复现冻结结果

先安装依赖。首次运行会从 Hugging Face 下载本地 embedding/reranker 模型。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

无精排基线：

```bash
EMBED_MODEL=BAAI/bge-small-zh-v1.5 python scripts/eval_retrieval.py \
  --vector-store faiss \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json \
  --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl \
  --backend all --ks 1,4 --embed-backend local --no-strict \
  --output-json data/eval/retrieval_synth_chunk_report.json
```

增加 BGE 精排：

```bash
EMBED_MODEL=BAAI/bge-small-zh-v1.5 python scripts/eval_retrieval.py \
  --vector-store faiss \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json \
  --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl \
  --backend all --ks 1,4 --embed-backend local --no-strict --rerank \
  --output-json data/eval/retrieval_synth_chunk_rerank_report.json
```

## 从语料重建索引

以下命令会重建 FAISS/BM25 索引：

```bash
EMBED_MODEL=BAAI/bge-small-zh-v1.5 python scripts/build_index.py \
  --input-dir data/uploads_eval \
  --vector-store faiss \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json \
  --embed-backend local
```

如果切块实现或参数发生变化，`chunk_id` 可能改变。此时必须重新核对 qrels，不能直接沿用冻结标注。

## 边界

- 数据集规模小且主题与本项目高度相关，只用于回归与相对比较。
- 主结果只使用 chunk-level qrels；旧的文档级标注不纳入主报告。
- 报告应同时注明模型、索引、是否精排、查询数和标注粒度，避免不同口径的数字混在一起。
