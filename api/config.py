"""配置：全部在调用时读环境变量，方便测试 monkeypatch 与线上覆盖。"""
import os


def max_upload_bytes() -> int:
    return int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)


def ask_timeout() -> float:
    return float(os.getenv("ASK_TIMEOUT_SEC", "120"))


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8501")
    return [o.strip() for o in raw.split(",") if o.strip()]


def tavily_api_key() -> str:
    """Tavily key：缺省或明显是占位符时，网页兜底视为不可用。"""
    return (os.getenv("TAVILY_API_KEY") or "").strip()


def web_search_available() -> bool:
    """真实 Tavily key 至少 20 位；.env 里 7 位的占位符会被判为不可用。"""
    return len(tavily_api_key()) >= 20


def default_enable_web_search() -> bool:
    """请求没显式传 enable_web_search 时，按服务端能力决定默认值。"""
    return web_search_available()
