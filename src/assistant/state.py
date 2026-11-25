import operator
from typing import Annotated
from typing_extensions import TypedDict

# 主图状态：贯穿整条链路，节点只返回自己要写的字段（增量更新）
class ResearcherState(TypedDict):
    user_instructions: str          # 用户的原始研究指令，由入口传入，全程只读
    research_queries: list[str]     # 问题拆解产出的子查询列表（generate_research_queries 写入）
    search_summaries: Annotated[list, operator.add]  # 各子查询回传的摘要；operator.add 表示并发分支的结果累加合并，不互相覆盖
    current_position: int           # 批次指针，每轮 +BATCH_SIZE，决定下一批取哪几条子查询
    final_answer: str               # 最终报告正文（generate_final_answer 写入）

# 入口 schema：只接受 user_instructions，其余字段不允许外部注入
class ResearcherStateInput(TypedDict):
    user_instructions: str

# 出口 schema：对外只暴露 final_answer
class ResearcherStateOutput(TypedDict):
    final_answer: str

# 子图状态：一次只管一条子查询，每个 Send 分支各自独立拥有一份
class QuerySearchState(TypedDict):
    query: str                      # 本条子查询
    web_search_results: list        # Tavily 兜底结果（仅在文档不相关且开启 web 时写入）
    retrieved_documents: list       # 本地向量库检索到的 Document 列表
    are_documents_relevant: bool    # 相关性判定结果，决定后续走摘要还是走网页
    search_summaries: list[str]     # 本子查询产出的摘要（单元素列表，回传主图时由 operator.add 拼接）

# 子图入口 schema：每个 Send 只传 query 进来
class QuerySearchStateInput(TypedDict):
    query: str

# 子图出口 schema：只把 query 和摘要回传给主图
class QuerySearchStateOutput(TypedDict):
    query: str
    search_summaries: list[str]
