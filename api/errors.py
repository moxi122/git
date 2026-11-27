"""统一错误结构：{"error": {"code", "message", "request_id"}}。

业务异常统一抛 AppError，由 main.py 的 handler 转成 JSON，
这样客户端只需要判断 code，不必解析 FastAPI 默认的 detail 文案。
"""


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def error_body(code: str, message: str, request_id: str | None = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id or "-",
        }
    }
