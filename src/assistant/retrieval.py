"""混合检索：BM25 关键词召回 + 向量语义召回，用 RRF 融合排名。

为什么这么设计：
1. BM25 的语料直接取自 Chroma 里的全部切片，不额外存一份索引文件，
   避免“向量库更新了、BM25 索引还是旧的”这种数据不同步问题；
2. 中文必须分词，jieba 切开后再算词频，否则 BM25 会退化成按单字匹配；
3. 融合用 RRF（Reciprocal Rank Fusion）：向量距离和 BM25 分数完全不同量纲，
   直接加权要反复调参，RRF 只看名次，换语料、换模型都不用重新调权重。
"""
import os
import re
import time
import hashlib
import logging
from collections import defaultdict

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.assistant.vector_db import (
    get_or_create_vector_db,
    list_all_documents,
    get_corpus_version,
)

logger = logging.getLogger(__name__)

# 召回宽、精排窄：向量和 BM25 各取 15 条候选，融合后只留 4 条给下游的“相关性判定”
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "15"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "15"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "4"))
RRF_K = int(os.getenv("RRF_K", "60"))

# 英文单词 / 数字整体保留，中文交给 jieba
_ALNUM_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

# 极简停用词：只去掉检索时几乎没有信息量的高频虚词，避免误伤专业词
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "等", "我", "你", "他", "它",
    "这", "那", "有", "无", "为", "对", "从", "到", "上", "下", "中",
    "一个", "什么", "怎么", "如何", "哪些",
}


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索词。

    分两趟处理，避免 jieba 把代码标识符切碎：
    1. 先用正则把英文/数字/下划线整体抽出来（batch_size、RAG、v1.5 这类保持完整）；
    2. 再让 jieba 切剩下的中文，只保留纯中文词。
    这是中文 BM25 最关键的一步：不分词等于按字匹配，词频统计几乎没有意义。
    """
    text = text.lower()
    tokens: list[str] = []

    # 第一趟：英文单词、数字、下划线标识符整体保留
    for token in _ALNUM_PATTERN.findall(text):
        if token not in _STOPWORDS:
            tokens.append(token)

    # 第二趟：jieba 切中文；英文片段已经在第一趟处理过，这里只收纯中文词
    for token in jieba.lcut(text):
        token = token.strip()
        if not token or token in _STOPWORDS:
            continue
        if all("\u4e00" <= ch <= "\u9fff" for ch in token):
            tokens.append(token)

    return tokens


# BM25 索引缓存：语料版本号变了（有新文档入库）就重建
_bm25_cache: dict = {"version": -1, "bm25": None, "documents": []}


def _rebuild_bm25() -> None:
    """从 Chroma 拉全量切片，分词后重建 BM25 索引。"""
    documents = list_all_documents()
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    _bm25_cache.update(version=get_corpus_version(), bm25=bm25, documents=documents)
    logger.info("BM25 索引已重建：%d 条切片", len(documents))


def _get_bm25() -> tuple[BM25Okapi | None, list[Document]]:
    """拿到当前有效的 BM25 索引，必要时重建。"""
    if _bm25_cache["version"] != get_corpus_version():
        _rebuild_bm25()
    return _bm25_cache["bm25"], _bm25_cache["documents"]


def vector_search(query: str, k: int = VECTOR_TOP_K) -> list[Document]:
    """向量召回：走 Chroma 的相似度检索。"""
    vectorstore = get_or_create_vector_db()
    return vectorstore.similarity_search(query, k=k)


def bm25_search(query: str, k: int = BM25_TOP_K) -> list[Document]:
    """BM25 召回：算出全部切片的分数后取前 k 条，分数为 0 的直接丢掉。"""
    bm25, documents = _get_bm25()
    if bm25 is None or not documents:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [documents[i] for i in ranked if scores[i] > 0]


def _document_key(doc: Document) -> str:
    """给切片算指纹：来源路径 + 正文，用于融合时去重。"""
    raw = f"{doc.metadata.get('source', '')}|{doc.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def rrf_fuse(
    ranked_lists: list[list[Document]],
    k: int = RRF_K,
    top_n: int = FINAL_TOP_N,
) -> list[Document]:
    """RRF 融合：每个文档得分 = Σ 1/(k + 名次)，按总分排序取前 top_n。

    只依赖名次、不依赖各家分数量纲，所以换 embedding、换分词方案都不用重新调权重。
    常数 k=60 沿用 RRF 原论文的取值，作用是压低头部名次的差距，让多路召回都有贡献。
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for documents in ranked_lists:
        for rank, doc in enumerate(documents, start=1):
            key = _document_key(doc)
            scores[key] += 1.0 / (k + rank)
            doc_map.setdefault(key, doc)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [doc_map[key] for key, _ in ordered]


def hybrid_retrieve(
    query: str,
    vector_k: int | None = None,
    bm25_k: int | None = None,
    top_n: int | None = None,
    enable_hybrid: bool | None = None,
) -> list[Document]:
    """混合检索入口：向量 + BM25 → RRF 融合。

    HYBRID_ENABLED=false 时退化成纯向量检索，行为和改造前一致，
    线上效果不对时可以一键降级，不用改代码。
    """
    vector_k = VECTOR_TOP_K if vector_k is None else vector_k
    bm25_k = BM25_TOP_K if bm25_k is None else bm25_k
    top_n = FINAL_TOP_N if top_n is None else top_n

    if enable_hybrid is None:
        enable_hybrid = os.getenv("HYBRID_ENABLED", "true").lower() not in {"false", "0", "no"}

    start = time.perf_counter()

    # 降级路径：只走向量，保持与旧版本一致
    if not enable_hybrid:
        documents = vector_search(query, k=top_n)
        logger.info("纯向量检索：%d 条，用时 %.0f ms", len(documents), (time.perf_counter() - start) * 1000)
        return documents

    vector_docs = vector_search(query, k=vector_k)

    # BM25 出任何问题都不能拖垮主链路，降级成纯向量
    try:
        keyword_docs = bm25_search(query, k=bm25_k)
    except Exception as exc:
        logger.warning("BM25 召回失败，降级为纯向量检索：%s", exc)
        keyword_docs = []

    fused = rrf_fuse([vector_docs, keyword_docs], top_n=top_n)
    logger.info(
        "混合检索：向量 %d 条 + BM25 %d 条 → 融合 %d 条，用时 %.0f ms",
        len(vector_docs),
        len(keyword_docs),
        len(fused),
        (time.perf_counter() - start) * 1000,
    )
    return fused
