"""FastAPI 应用工厂。

路由只做"参数校验 + 调用 + 包装"，业务仍在 src/assistant 里，
所以这个后端是加在现有 RAG 引擎外面的一层薄门面，而不是第二套实现。
"""
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv

# 加载项目根目录的 .env：README 让直接用 uvicorn 启动，不加载的话
# DEEPSEEK_API_KEY / ASK_TIMEOUT_SEC 这些配置在 API 进程里全是空值
load_dotenv()

from api.config import cors_origins, web_search_available
from api.errors import AppError, error_body
from api.routes import ask, health, ingest, retrieve

logger = logging.getLogger("api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local RAG Researcher API",
        version="0.1.0",
        description="Agentic RAG 研究代理的 HTTP 门面：检索 / 问答 / 流式 / 入库。",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        request.state.request_id = rid
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = rid
        logger.info(
            "%s %s -> %s %.1fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            rid,
        )
        return response

    def _rid(request: Request) -> str:
        return getattr(request.state, "request_id", "-")

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, _rid(request)),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        msg = first.get("msg", "参数校验失败")
        message = f"{loc}: {msg}" if loc else str(msg)
        return JSONResponse(
            status_code=422,
            content=error_body("VALIDATION_ERROR", message, _rid(request)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(
            exc.status_code, "HTTP_ERROR"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, str(exc.detail), _rid(request)),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception("未捕获异常 rid=%s", _rid(request))
        return JSONResponse(
            status_code=500,
            content=error_body("INTERNAL_ERROR", "服务内部错误，请查日志", _rid(request)),
        )

    app.include_router(health.router, prefix="/v1")
    app.include_router(retrieve.router, prefix="/v1")
    app.include_router(ask.router, prefix="/v1")
    app.include_router(ingest.router, prefix="/v1")

    # 启动自检：Tavily key 缺失/像占位符时，/ask 不传 enable_web_search 就默认关闭兜底，
    # 避免默认请求在检索落空时被一个无效 key 拖成 502；显式传 true 仍然会尝试
    if not web_search_available():
        logger.warning(
            "TAVILY_API_KEY 未配置或明显无效：enable_web_search 默认关闭，"
            "调用方显式传 true 时才会尝试网页兜底"
        )

    return app


app = create_app()
