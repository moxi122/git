# 🚀 **基于 DeepSeek R1 与 LangGraph 的本地 RAG 研究员**

### 👉 **[内附分步教程：从零搭建你自己的本地 RAG 研究员！](https://dev.to/kaymen99/build-your-own-local-rag-researcher-with-deepseek-r1-11m) 🚀**

我用 **LangGraph** 和一个跑在 **Ollama** 上的本地 **DeepSeek R1** 模型，做了一个**本地自适应 RAG 研究代理**。它像一位深度研究员，会按用户指令去收集、分析并总结信息。

<div align="center">
  <img src="https://github.com/user-attachments/assets/5dc34341-3a2f-461c-b66d-46b134fe5bd9" alt="Demo of Local RAG Researcher with LangGraph & DeepSeek">
</div>

## **工作流程**

1. **生成研究查询** – 接收用户输入，把它拆成若干条相关的研究问题，以便找到最有用的信息。

2. **检索文档** – 在本地 **Chroma 向量库**中检索与该查询相关的文档。

3. **评估相关性** – 把每份文档与原始查询逐一比对，确认它是否包含有意义且准确的信息。

4. **按需扩展检索** – 如果检索到的文档不充分或不相关，代理可以**联网搜索**补充来源。

5. **汇总发现** – 信息收集齐全后，对数据做处理并提炼关键洞察。

6. **生成最终报告** – 把汇总结果交给**撰写代理**，按预设格式整理成一份**结构完整、排版规范的报告**。

这套流程让研究过程**高效且自适应**：既保证输出质量与相关性，又尽量避免引入无用或低价值的信息。

## **核心特性**

- **本地文档动态检索** – 从你的内部文档中高效检索相关信息。
- **深度洞察提取** – 借助 **DeepSeek R1** 的推理能力评估、分析并提取文档中最有价值的洞察。
- **实时联网搜索** – 本地文档不足时，通过 **[Tavily API](https://tavily.com/)** 访问在线来源扩展研究。
- **结构化报告生成** – 依据你预设的报告模板产出格式规范的报告。

## **混合检索（BM25 + 向量 + RRF 融合）**

本地知识库的召回不是单纯走向量，而是两路召回后融合：

```
query
 ├─ 向量召回：千问 text-embedding-v3（DashScope OpenAI 兼容端点）+ Chroma   top 15
 ├─ 关键词召回：jieba 分词 + BM25Okapi                                      top 15
 ├─ RRF 融合：score = Σ 1/(60 + 名次)                                        → top 4
 └─ 交给下游的“相关性判定 → 摘要”
```

几个设计点：

- **BM25 语料直接取自 Chroma**，不单独维护索引文件，避免"向量库更新了、关键词索引还是旧的"。
- **中文必须分词**：jieba 切开后再统计词频，否则 BM25 会退化成按单字匹配。
- **用 RRF 而不是分数加权**：余弦距离和 BM25 分数不同量纲，加权要反复调参；RRF 只看名次，换模型、换语料都不用重调。
- **可一键降级**：`HYBRID_ENABLED=false` 时退回纯向量检索，行为和改造前完全一致。

关键参数都在 `.env` 里：`VECTOR_TOP_K`、`BM25_TOP_K`、`FINAL_TOP_N`、`RRF_K`。
检索耗时会打在日志里（`混合检索：向量 x 条 + BM25 y 条 → 融合 z 条，用时 N ms`），方便做前后对比。

## 系统流程图

系统的完整流程如下：

<div align="center">
  <img src="https://github.com/user-attachments/assets/5e06e948-c853-47d1-b25e-e3c5ca96b60d" alt="Langgraph Local Deepseek RAG researcher">
</div>


## 技术栈
- **[Ollama](https://ollama.com/)**：本地运行 DeepSeek R1 模型。
- **[LangGraph](https://www.langchain.com/langgraph)**：构建 AI 代理并定义研究员的工作流。
- **[ChromaDB](https://docs.trychroma.com/)**：用于 RAG 检索的本地向量数据库。
- **[千问 text-embedding-v3](https://help.aliyun.com/zh/model-studio/)**：知识库向量化（DashScope OpenAI 兼容端点）。
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25) + jieba**：关键词召回与中文分词。
- **[Streamlit](https://docs.streamlit.io/)**：提供与研究员交互的界面。
- **[Tavily](https://tavily.com/)**：用于联网搜索。

## 如何运行
### 环境要求
请确认已安装以下内容：
- Python 3.9+
- Ollama
- 用于联网搜索的 Tavily API Key
- 所需 Python 依赖（见 `requirements.txt`）

### 安装步骤
#### 克隆仓库
```bash
git clone https://github.com/kaymen99/local-rag-researcher-deepseek
cd local-rag-researcher-deepseek
```

#### 创建并激活虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows 下用 `venv\Scripts\activate`
```

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 配置环境变量
在根目录创建 `.env` 文件，填入所需凭证：

```ini
# Tavily 搜索工具的 API Key（可选）
TAVILY_API_KEY="your-api-key"
```

## 启动应用
### 第 1 步：安装并运行 Ollama

按官方说明[安装 Ollama](https://ollama.com/download)，然后拉取 DeepSeek R1 模型（本项目用的是 7b 版本，你也可以选择[其他可用模型](https://ollama.com/library/deepseek-r1)）：

```bash
ollama pull deepseek-r1:7b
```

### 第 2 步：启动 Streamlit 应用

执行以下命令启动界面：

```bash
streamlit run app.py
```

### 第 3 步：在 LangGraph Studio 中可视化（可选）

由于本项目基于 LangGraph 构建，你可以用 **LangGraph Studio** 查看代理的工作流。执行以下命令：

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev
```

## 自定义配置
### 修改报告结构
- 在 `report_structures` 目录下新增自定义结构文件。
- 在界面中选择想用的结构。

### 使用外部 LLM 服务商

默认情况下，研究员完全本地运行，使用 **Ollama** 上的 **DeepSeek R1** 模型。不过，如果你更想用云端 LLM 服务商（比如**云端 DeepSeek R1**、**OpenAI GPT-4o** 或 **OpenAI o1**），按以下步骤操作：

1. **修改代码**：
   - 打开 `src/assistant/graph.py`。
   - 注释掉调用 Ollama 模型的那段代码。
   - 取消启用外部 LLM 调用那段代码的注释。
   - `invoke_llm` 走的是 **[OpenRouter](https://openrouter.ai)**，它提供对多家 LLM 的访问，可在其[模型列表](https://openrouter.ai/models)里挑选中意的模型。🚀
   - 如果你愿意，也可以改造 `invoke_llm`，直接对接某一家服务商而不经过 **OpenRouter**。

2. **配置 API Key**：
   - 在[这里](https://openrouter.ai/settings/keys)获取 OpenRouter API Key。
   - 按下面的格式写进 `.env` 文件：

     ```env
     OPENROUTER_API_KEY=your_openai_key
     ```

## **📚 延伸阅读与资源**

* LangChain：用 DeepSeek-R1 构建完全本地的「deep researcher」[见](https://www.youtube.com/watch?v=sGUjmyfof4Q)

* LangChain：从零构建基于 Ollama 的完全本地研究助手[见](https://www.youtube.com/watch?v=XGuTzHoqlj8)

* LangGraph 模板：多代理 RAG 研究[见](https://www.youtube.com/watch?v=JLDLANs_m_w)

* LangGraph 自适应 RAG 实现[见](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag_local.ipynb)

## 参与贡献
欢迎贡献！有任何改动，请开 issue 或直接提交 pull request。

## 联系方式
如有问题或建议，欢迎邮件联系 aymenMir1001@gmail.com。

---

## 本项目当前状态（与上游原文不同的地方）

上面正文是上游 README 的忠实翻译，但它描述的默认运行方式已经被本地改动覆盖了，实际以本节为准：

- **LLM 默认走 DeepSeek 官方 API**，不再是 Ollama。`.env` 里配 `DEEPSEEK_API_KEY` 即可；Ollama 调用保留在注释里，改一行就能切回本地。
- **模型分工**：`deepseek-chat` 负责结构化输出（拆查询、相关性判定），`deepseek-reasoner` 负责摘要与成稿；成稿显式设了 `max_tokens=8000`，避免默认 4096 上限截断长报告。
- **注释与 prompt 已全部中文化**：`src/assistant/prompts.py` 四段 prompt、各文件里的英文注释均为中文。
- **`parse_output` 做了双模式兼容**：带不带 `<think>` 标签都能解析。原版硬取 `<think>`，换成 API 模型后这一处会直接崩。
- **新增 `.env` 与 `.gitignore`**：密钥不进版本库。
- **原文里的 `assistant/graph.py` 实际路径是 `src/assistant/graph.py`**，译文已按真实路径修正。
- **英文原版保留在 `README.en.md`**，方便对照。
- **检索层升级为混合召回（BM25 + 向量 + RRF）**：新增 `src/assistant/retrieval.py`，`retrieve_rag_documents` 节点改为调用 `hybrid_retrieve()`；BM25 语料直接从 Chroma 读取，入库后自动失效重建。
- **embedding 换成千问 `text-embedding-v3`**：走 DashScope 的 OpenAI 兼容端点（`QWEN_API_KEY`），不再依赖本地 HuggingFace 模型。
- **切块方式改为中文友好的定长切分**：原 `SemanticChunker` 会对每个句子调一次 embedding，换成 API 后请求量不可接受，改为 `RecursiveCharacterTextSplitter`（按 `。！？；` 断句，800 字 / 150 重叠，可用 `CHUNK_SIZE`、`CHUNK_OVERLAP` 调整）。

还没打通的地方（与 LLM 无关，属环境准备）：

1. `./files` 目录不存在，首次建库会卡在 `DirectoryLoader("./files")`
2. `unstructured` 未安装（`requirements.txt` 里列了但没装上），`DirectoryLoader` 默认加载器依赖它
3. 首次建库需要配置千问 API Key；向量维度由 `text-embedding-v3` 决定，换模型必须删掉 `database/` 重建

