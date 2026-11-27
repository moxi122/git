import os
import logging
from functools import lru_cache

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "database")

# 中文切块用的分隔符：优先按段落切，其次按句号/叹号/问号/分号，最后才按字符硬切
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 语料版本号：每次入库 +1，retrieval 里的 BM25 缓存靠它判断要不要重建
_corpus_version = 0


def get_corpus_version() -> int:
    """返回当前语料版本号（同进程内有效，重启后从 0 开始）。"""
    return _corpus_version


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """千问（DashScope）向量模型，走 OpenAI 兼容端点。

    两个关键参数：
    - check_embedding_ctx_length=False：OpenAIEmbeddings 默认会用 tiktoken 把文本
      切成 token 数组再发请求，那是 OpenAI 的私有协议，DashScope 兼容端点只认原始
      字符串，所以必须关掉，否则请求会 400。
    - chunk_size=10：DashScope 的文本向量接口单次最多 10 条，超了会报错。
    """
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 QWEN_API_KEY，请在 .env 里配置千问（DashScope）的 API Key")

    return OpenAIEmbeddings(
        model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3"),
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        check_embedding_ctx_length=False,
        chunk_size=int(os.getenv("QWEN_EMBEDDING_BATCH", "10")),
        max_retries=3,
    )


def split_documents(documents):
    """把原始文档切成检索切片。

    改造说明：原版用 SemanticChunker 按语义断点切分，但它会对每个句子调一次
    embedding。embedding 换成千问 API 之后，一份几十页的 PDF就是几百次请求，
    既慢又贵还容易被限流，所以改成“定长 + 中文标点优先”的切分方式。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )
    return splitter.split_documents(documents)


def _load_documents_from_folder(folder: str = "./files") -> list[Document]:
    """按扩展名逐个加载文档，不依赖 unstructured。

    原版直接用 DirectoryLoader 的默认加载器（UnstructuredFileLoader），它依赖
    unstructured 包；本项目没装这个包，首次建库会直接报错。这里按扩展名显式指定
    加载器：txt/md 用 TextLoader，pdf 用 PDFPlumberLoader。
    """
    os.makedirs(folder, exist_ok=True)

    documents: list[Document] = []
    loaders = [
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
        ("**/*.pdf", PDFPlumberLoader, {}),
    ]

    for pattern, loader_cls, loader_kwargs in loaders:
        try:
            loader = DirectoryLoader(
                folder,
                glob=pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
            )
            documents.extend(loader.load())
        except Exception as exc:
            # 单个格式加载失败不影响其它格式，记日志继续
            logger.warning("加载 %s 失败：%s", pattern, exc)

    return documents


def get_or_create_vector_db() -> Chroma:
    """获取向量库：已有库直接加载，没有就现场建库。"""
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # 库已存在（目录非空）：直接加载，不再重复做 embedding
        return Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    # 库不存在：读取 ./files 目录并现场建库
    docs = _load_documents_from_folder()
    split = split_documents(docs)

    # 目录里没有可读文档时，返回一个空库，避免上层拿到 None 再去调方法报错
    if not split:
        return Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_PATH)

    return Chroma.from_documents(split, embeddings, persist_directory=VECTOR_DB_PATH)


def add_documents(documents):
    """把新文档追加进已有的向量库，并让 BM25 索引失效重建。

    参数：
        documents: 待写入向量库的文档列表
    """
    global _corpus_version

    embeddings = get_embeddings()
    split = split_documents(documents)
    if not split:
        return get_or_create_vector_db()

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # 库已存在：增量追加
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        vectorstore.add_documents(split)
    else:
        # 库不存在：新建
        vectorstore = Chroma.from_documents(split, embeddings, persist_directory=VECTOR_DB_PATH)

    # 语料变了，BM25 缓存必须重建，否则新文档既搜不到向量、也搜不到关键词
    _corpus_version += 1
    logger.info("已入库 %d 条切片，语料版本更新为 %d", len(split), _corpus_version)

    return vectorstore


def list_all_documents() -> list[Document]:
    """取出库里全部切片，供 BM25 建索引使用。

    直接复用 Chroma 作为唯一数据源，不再单独维护一份 BM25 语料文件，
    从根上避免“向量库和关键词索引不同步”。
    """
    vectorstore = get_or_create_vector_db()
    data = vectorstore.get(include=["documents", "metadatas"])

    contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(contents, metadatas)
    ]
