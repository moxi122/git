import asyncio
import json
import threading
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.config import ask_timeout
from api.deps import get_graph
from api.errors import AppError
from api.schemas import AskRequest, AskResponse, StepOut

router = APIRouter(tags=["research"])


def _run_graph(graph, request: AskRequest):
    """同步跑完整条 LangGraph，收集节点名 + 截断后的增量。"""
    initial_state = {
        "question": request.question,
        "max_queries": request.max_queries,
        "enable_web_search": request.enable_web_search,
    }
    steps: list[dict] = []
    answer = ""
    for chunk in graph.stream(initial_state):
        for node, value in chunk.items():
            steps.append({"node": node, "payload": str(value)[:500]})
            if node == "generate_final_answer" and isinstance(value, dict):
                answer = value.get("final_answer", "")
    return steps, answer


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, graph=Depends(get_graph)):
    started = time.perf_counter()
    try:
        steps, answer = await asyncio.wait_for(
            asyncio.to_thread(_run_graph, graph, request),
            timeout=ask_timeout(),
        )
    except asyncio.TimeoutError:
        raise AppError("ASK_TIMEOUT", f"研究流程超过 {ask_timeout():.0f}s 未完成", 504)
    except Exception as exc:  # LLM 网关、检索、结构化输出解析失败都归到这里
        raise AppError("ASK_FAILED", f"研究流程执行失败: {exc}", 502)

    if not answer:
        raise AppError("EMPTY_ANSWER", "流程结束但没有产出 final_answer", 502)

    return AskResponse(
        answer=answer,
        steps=[StepOut(**step) for step in steps],
        took_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post("/ask/stream")
async def ask_stream(request: AskRequest, graph=Depends(get_graph)):
    """SSE 推送：graph.stream 在工作线程里跑，事件通过 asyncio.Queue 回传。

    LangGraph 是同步迭代器，直接把 for 循环写在 async 生成器里会阻塞事件循环，
    所以必须丢到线程池；队列 + call_soon_threadsafe 是跨线程安全回传的标准做法。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker() -> None:
        try:
            initial_state = {
                "question": request.question,
                "max_queries": request.max_queries,
                "enable_web_search": request.enable_web_search,
            }
            for chunk in graph.stream(initial_state):
                loop.call_soon_threadsafe(queue.put_nowait, ("node", chunk))
        except Exception as exc:  # 错误也要通过流告诉客户端，而不是 200 后静默断开
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        yield "event: start\ndata: {}\n\n"
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": ping\n\n"  # 心跳，防止反代把长连接掐掉
                continue

            if kind == "end":
                break
            if kind == "error":
                data = json.dumps({"message": payload}, ensure_ascii=False)
                yield f"event: error\ndata: {data}\n\n"
                break

            for node, value in payload.items():
                data = json.dumps(
                    {"node": node, "payload": str(value)[:500]}, ensure_ascii=False
                )
                yield f"event: node\ndata: {data}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
