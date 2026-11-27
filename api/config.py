"""配置：全部在调用时读环境变量，方便测试 monkeypatch 与线上覆盖。"""
import os


def max_upload_bytes() -> int:
    return int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)


def ask_timeout() -> float:
    return float(os.getenv("ASK_TIMEOUT_SEC", "120"))


def cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8501")
    return [o.strip() for o in raw.split(",") if o.strip()]
