"""API 层契约测试。

只验证"接口行为"：状态码、返回结构、错误码；
不验证答案质量、召回效果、幻觉——那些属于评测，不属于功能测试。
所有外部依赖（LLM、embedding、向量库）都用 fake 替换，保证测试确定性。
"""
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.deps import get_graph, get_ingestor, get_retriever
from api.main import create_app


class FakeGraph:
    """假的 LangGraph：只按节点的增量输出，不调 LLM。"""

    def stream(self, state, config=None):
        yield {"generate_research_queries": {"research_queries": ["子查询1", "子查询2"]}}
        yield {"search_and_summarize_query": {"search_summaries": ["摘要1"]}}
        yield {"generate_final_answer": {"final_answer": "这是最终报告正文"}}


class SlowGraph:
    """用来触发 /ask 超时分支。"""

    def __init__(self, seconds: float = 2.0):
        self.seconds = seconds

    def stream(self, state, config=None):
        import time

        time.sleep(self.seconds)
        yield {"generate_final_answer": {"final_answer": "超时前不该拿到"}}
        return


class ExplodingGraph:
    def stream(self, state, config=None):
        raise RuntimeError("LLM 网关异常")
        yield  # pragma: no cover


class FakeIngestor:
    """记录收到的文件名，验证适配器把 UploadFile 转成了带 name/getvalue 的对象。"""

    def __init__(self):
        self.received = []

    def __call__(self, uploads):
        self.received.extend([u.name for u in uploads])
        return True


@pytest.fixture
def ingestor():
    return FakeIngestor()


@pytest.fixture
def client(ingestor):
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: FakeGraph()
    app.dependency_overrides[get_retriever] = lambda: (
        lambda query, top_k: [
            Document(page_content="切片内容 A", metadata={"source": "a.md"}),
            Document(page_content="切片内容 B", metadata={"source": "b.md"}),
        ][:top_k]
    )
    app.dependency_overrides[get_ingestor] = lambda: ingestor
    with TestClient(app) as c:
        yield c


# ---------- 正常用例 ----------

def test_health_returns_ok_and_request_id(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.headers.get("X-Request-ID")


def test_retrieve_returns_documents_with_source(client):
    resp = client.post("/v1/retrieve", json={"query": "混合检索怎么做", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["documents"]) == 2
    assert body["documents"][0]["source"] == "a.md"
    assert body["documents"][0]["content"] == "切片内容 A"


def test_ask_returns_answer_and_steps(client):
    resp = client.post("/v1/ask", json={"question": "介绍一下 Agentic RAG"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "这是最终报告正文"
    assert [s["node"] for s in body["steps"]] == [
        "generate_research_queries",
        "search_and_summarize_query",
        "generate_final_answer",
    ]


def test_ask_stream_emits_sse_node_events(client):
    with client.stream("POST", "/v1/ask/stream", json={"question": "流式测试"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        payload = "".join(resp.iter_text())
    assert "event: node" in payload
    assert "generate_research_queries" in payload
    assert "event: done" in payload


def test_ingest_markdown_succeeds(client, ingestor):
    resp = client.post(
        "/v1/ingest",
        files={"file": ("note.md", b"# title\n\ncontent", "text/markdown")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"filename": "note.md", "ingested": True}
    assert ingestor.received == ["note.md"]


# ---------- 边界用例 ----------

@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "太长" * 3000},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": 99},
    ],
)
def test_retrieve_rejects_bad_input(client, payload):
    resp = client.post("/v1/retrieve", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "ok", "max_queries": 0},
        {"question": "ok", "max_queries": 11},
    ],
)
def test_ask_rejects_bad_input(client, payload):
    resp = client.post("/v1/ask", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


# ---------- 异常用例 ----------

def test_ask_timeout_returns_504(monkeypatch):
    monkeypatch.setenv("ASK_TIMEOUT_SEC", "0.5")
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: SlowGraph(seconds=2)
    with TestClient(app) as c:
        resp = c.post("/v1/ask", json={"question": "会很慢"})
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "ASK_TIMEOUT"


def test_ask_llm_failure_returns_502():
    app = create_app()
    app.dependency_overrides[get_graph] = lambda: ExplodingGraph()
    with TestClient(app) as c:
        resp = c.post("/v1/ask", json={"question": "会炸"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "ASK_FAILED"


def test_retrieve_backend_failure_returns_502():
    def broken_retriever(query, top_k):
        raise RuntimeError("向量库不可用")

    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: broken_retriever
    with TestClient(app) as c:
        resp = c.post("/v1/retrieve", json={"query": "任意"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "RETRIEVE_FAILED"


def test_ingest_rejects_unsupported_extension(client):
    resp = client.post(
        "/v1/ingest",
        files={"file": ("virus.exe", b"MZ", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNSUPPORTED_FILE"


def test_ingest_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.000001")  # 约 1 字节
    resp = client.post(
        "/v1/ingest",
        files={"file": ("big.md", b"x" * 10, "text/markdown")},
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_unknown_route_returns_structured_404(client):
    resp = client.get("/v1/nope")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["request_id"]
