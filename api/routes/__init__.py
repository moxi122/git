"""路由注册入口：新增端点只改这里。"""
from api.routes import ask, health, ingest, retrieve

__all__ = ["ask", "health", "ingest", "retrieve"]
