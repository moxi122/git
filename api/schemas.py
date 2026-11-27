"""请求/响应模型：这里就是接口契约的可执行版本。"""
from pydantic import BaseModel, Field, field_validator


class RetrieveRequest(BaseModel):
    query: str = Field(..., max_length=5000, description="检索语句")
    top_k: int = Field(4, ge=1, le=20, description="返回切片数量")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value


class DocumentOut(BaseModel):
    content: str
    source: str
    score: float | None = None


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    documents: list[DocumentOut]


class AskRequest(BaseModel):
    question: str = Field(..., max_length=5000, description="研究问题")
    max_queries: int = Field(5, ge=1, le=10, description="最多拆出多少条子查询")
    enable_web_search: bool = Field(True, description="是否允许 Tavily 网页兜底")

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class StepOut(BaseModel):
    node: str
    payload: str


class AskResponse(BaseModel):
    answer: str
    steps: list[StepOut]
    took_ms: int


class IngestResponse(BaseModel):
    filename: str
    ingested: bool
