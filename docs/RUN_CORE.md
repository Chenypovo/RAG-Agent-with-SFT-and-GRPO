# 跑通记忆增强 RAG Core

这份指南用于在本地跑通带长期记忆的 RAG Agent：助手从多轮对话中沉淀事实，并在后续会话中召回。

## 1. 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

全量依赖包含 reranker、OCR 和多模态组件，安装体积较大，但可保证完整测试集和所有入口使用同一套环境。

## 2. 配置 .env

```bash
cp .env.example .env
```

填入你的 OpenAI 兼容服务。下面以兼容接口为例，模型名必须与服务商实际支持的名称一致：

```dotenv
LLM_PROVIDER=openai_compatible
LLM_MODEL=glm-4-flash
OPENAI_COMPAT_API_KEY=你的key
OPENAI_COMPAT_BASE_URL=你的base_url

# 默认使用本地 embedding，首次运行会下载模型
EMBED_PROVIDER=local
EMBED_MODEL=BAAI/bge-small-zh-v1.5
EMBED_DEVICE=cpu
```

## 3. 先跑单测（确认 core 逻辑健康，无需联网）

```bash
python -m pytest -q
```

期望命令以退出码 `0` 结束。测试数量会随功能变化，因此不在文档中硬编码。

## 4.（可选）建文档索引

有文档证据时回答更像 RAG；没有也能跑（纯记忆模式）。

```bash
# 把文件放进 data/uploads/ 后：
python scripts/build_index.py \
  --input-dir data/uploads \
  --vector-store lancedb \
  --lancedb-uri data/index/lancedb \
  --bm25-path data/index/bm25.json \
  --embed-backend local
```

## 5. 跑对话 core（重点）

```bash
python scripts/chat_demo.py
```

建议这样验证"长期记忆"确实生效：

1. 先**陈述事实**（会被写入记忆）：
   - `演示机器人的名字是阿尔法，它正在整理示例文档`
   - 观察输出末尾的 `↳ memory updated: add:...`
2. **换个话题**问一句无关的，再回来问：
   - `我在读什么专业？` / `我最近在做什么项目？`
   - 观察 `↳ recalled about you:` 是否召回了之前说的事实，回答是否用上了。
3. 输入 `:mem` 查看当前所有已存记忆。
4. 退出再进来 `python scripts/chat_demo.py`，再问一次——记忆应**跨会话保留**（存在 `data/memory/`）。

## 期望看到的现象

- 陈述事实那轮：`↳ memory updated: add:...`
- 后续相关提问：`↳ recalled about you: • ...` 并且回答个性化
- `:mem` 能列出结构化事实
- 重启后记忆仍在（`data/memory/memory.db` + `mem_index.json`）

## 6. 常见问题

- 启动时提示缺少 key/base URL：检查 `.env` 中 `OPENAI_COMPAT_API_KEY` 和 `OPENAI_COMPAT_BASE_URL`。
- 首次本地 embedding 较慢：需要下载 `BAAI/bge-small-zh-v1.5`，后续会复用本地缓存。
- 文档检索始终为空：确认已执行建库命令，且 `data/index/lancedb/` 与 `data/index/bm25.json` 存在。
- 记忆数据位于 `data/memory/`，默认不会提交到 Git。

> 当前服务只面向本地单用户演示；尚未实现鉴权、用户隔离和限流，不要直接作为公网多用户服务部署。
