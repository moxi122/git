import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.retrieval import hybrid_retrieve
from src.assistant.state import ResearcherState, ResearcherStateInput, ResearcherStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput
from src.assistant.prompts import RESEARCH_QUERY_WRITER_PROMPT, RELEVANCE_EVALUATOR_PROMPT, SUMMARIZER_PROMPT, REPORT_WRITER_PROMPT
from src.assistant.utils import format_documents_with_metadata, invoke_deepseek, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries

# 每批并行处理的子查询条数，同时也是 current_position 每轮的推进步长
# 按机器性能调整：调大并发更高，同时压给 Ollama 的请求也更多
BATCH_SIZE = 3

# 主图节点 1：把用户指令拆成 N 条子查询
# 读：state["user_instructions"]、config["configurable"]["max_search_queries"]
# 写：state["research_queries"]
def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    print("--- Generating research queries ---")
    user_instructions = state["user_instructions"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    
    query_writer_prompt = RESEARCH_QUERY_WRITER_PROMPT.format(
        max_queries=max_queries,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    )
    
    # 主路径：DeepSeek API + deepseek-chat（结构化输出必须用 chat，reasoner 不支持 JSON 约束）
    # 想换回本地 Ollama：把下面这行改成 invoke_ollama(model='deepseek-r1:7b', ...)
    result = invoke_deepseek(
        model='deepseek-chat',
        system_prompt=query_writer_prompt,
        user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
        output_format=Queries
    )
    
    # 方案 B：改用 OpenRouter 调外部模型（GPT-4o / Claude / DeepSeek R1…），取消注释即可切换
    # result = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=query_writer_prompt,
    #     user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
    #     output_format=Queries
    # )

    return {"research_queries": result.queries}

# 注意：下面是同名函数的早期草稿，会被第 56 行那份重新定义覆盖，实际不生效
def search_queries(state: ResearcherState):
    # 启动每条子查询的检索（真正扇出交给 initiate_query_research）
    print("--- Searching queries ---")
    pass

# 注意：这份 initiate_query_research 同样被后面的批次版本覆盖，实际不生效
def initiate_query_research(state: ResearcherState):
    # 用 Send 把每条子查询并发扇出到 search_and_summarize_query 子图
    return [
        Send("search_and_summarize_query", {"query": s})
        for s in state["research_queries"]
    ]

# 主图节点 2：只推进批次指针，不做检索
# 读：state["current_position"]（缺省 0）
# 写：state["current_position"]（+BATCH_SIZE）
def search_queries(state: ResearcherState):
    # 启动每条子查询的检索（真正扇出由 initiate_query_research 完成）
    print("--- Searching queries ---")
    # 取当前处理位置，缺失时按 0 处理
    current_position = state.get("current_position", 0)

    return {"current_position": current_position + BATCH_SIZE}

# 主图条件边：决定“继续下一批”还是“进入成稿”
# 唯一的循环终止条件：批次指针 >= 子查询总数
def check_more_queries(state: ResearcherState) -> Literal["search_queries", "generate_final_answer"]:
    """判断队列里是否还有未处理的子查询。"""
    current_position = state.get("current_position", 0)
    if current_position < len(state["research_queries"]):
        return "search_queries"
    return "generate_final_answer"

# 主图条件边（第 229 行挂载的那一个）：把本批子查询逐条 Send 出去
# 读：state["research_queries"]、state["current_position"]
# 返回：Send 列表（每条子查询一份独立的子图状态）
def initiate_query_research(state: ResearcherState):
    # 取下一批子查询
    queries = state["research_queries"]
    current_position = state["current_position"]
    batch_end = min(current_position, len(queries))
    current_batch = queries[current_position - BATCH_SIZE:batch_end]

    # 这一批逐条扇出
    return [
        Send("search_and_summarize_query", {"query": s})
        for s in current_batch
    ]

# 子图节点 1：从本地向量库检索
# 读：state["query"]
# 写：state["retrieved_documents"]
def retrieve_rag_documents(state: QuerySearchState):
    """从本地知识库检索文档：BM25 + 向量混合召回，RRF 融合后取前 4 条。

    具体参数（各路召回条数、融合保留条数）在 retrieval.py 里用环境变量控制，
    这里只负责把 query 传进去，节点读写的 state 字段保持不变。
    """
    print("--- Retrieving documents ---")
    query = state["query"]
    documents = hybrid_retrieve(query)

    return {"retrieved_documents": documents}

# 子图节点 2：判定检索结果是否足以回答该子查询
# 读：state["query"]、state["retrieved_documents"]
# 写：state["are_documents_relevant"]
def evaluate_retrieved_documents(state: QuerySearchState):
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]
    evaluation_prompt = RELEVANCE_EVALUATOR_PROMPT.format(
        query=query,
        documents=format_documents_with_metadata(retrieved_documents)
    )
    
    # 主路径：DeepSeek API + deepseek-chat（结构化输出）
    # 想换回本地 Ollama：把下面这行改成 invoke_ollama(model='deepseek-r1:7b', ...)
    evaluation = invoke_deepseek(
        model='deepseek-chat',
        system_prompt=evaluation_prompt,
        user_prompt=f"Evaluate the relevance of the retrieved documents for this query: {query}",
        output_format=Evaluation
    )
    
    # 方案 B：改用 OpenRouter 调外部模型（GPT-4o / Claude / DeepSeek R1…），取消注释即可切换
    # evaluation = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=evaluation_prompt,
    #     user_prompt=f"Evaluate the relevance of the retrieved documents for this query: {query}",
    #     output_format=Evaluation
    # )

    return {"are_documents_relevant": evaluation.is_relevant}

# 子图三岔路由：相关→直接摘要；不相关且开了 web→Tavily 兜底；不相关且没开 web→直接丢弃该子查询（无日志）
def route_research(state: QuerySearchState, config: RunnableConfig) -> Literal["summarize_query_research", "web_research", "__end__"]:
    """按“文档是否相关”做三岔路由。"""

    if state["are_documents_relevant"]:
        return "summarize_query_research"
    elif config["configurable"].get("enable_web_search", False):
        return "web_research"
    else:
        print("Skipping query due to irrelevant documents and web search disabled.")
        return "__end__"

# 子图节点 3（仅兜底分支）：网页检索
# 读：state["query"]
# 写：state["web_search_results"]
def web_research(state: QuerySearchState):
    print("--- Web research ---")
    output = tavily_search(state["query"])
    search_results = output["results"]

    return {"web_search_results": search_results}

# 子图节点 4：生成单条子查询的摘要
# 读：state["query"]、state["are_documents_relevant"]、state["retrieved_documents"] 或 state["web_search_results"]
# 写：state["search_summaries"]（单元素列表，回传主图后由 operator.add 拼接）
def summarize_query_research(state: QuerySearchState):
    query = state["query"]

    information = None
    if state["are_documents_relevant"]:
        # 文档相关：用本地检索到的文档
        information = state["retrieved_documents"]
    else:
        # 文档不相关：用网页结果；web 未开启时根本走不到这里，上一跳已丢弃
        information = state["web_search_results"]

    summary_prompt = SUMMARIZER_PROMPT.format(
        query=query,
        docmuents=information
    )
    
    # 主路径：DeepSeek API + deepseek-reasoner（自由文本，不需要 JSON 约束）
    # 想换回本地 Ollama：把下面这行改成 invoke_ollama(model='deepseek-r1:7b', ...)
    summary = invoke_deepseek(
        model='deepseek-reasoner',
        system_prompt=summary_prompt,
        user_prompt=f"Generate a research summary for this query: {query}"
    )
    # 去掉思维链，只保留正式回答（API 与 Ollama 两种输出格式都兼容）
    summary = parse_output(summary)["response"]
    
    # 方案 B：改用 OpenRouter 调外部模型（GPT-4o / Claude / DeepSeek R1…），取消注释即可切换
    # summary = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=summary_prompt,
    #     user_prompt=f"Generate a research summary for this query: {query}"
    # )

    return {"search_summaries": [summary]}

# 主图节点 3：把所有摘要拼成最终报告
# 读：state["user_instructions"]、state["search_summaries"]、config["configurable"]["report_structure"]
# 写：state["final_answer"]
def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    print("--- Generating final answer ---")
    report_structure = config["configurable"].get("report_structure", "")
    answer_prompt = REPORT_WRITER_PROMPT.format(
        instruction=state["user_instructions"],
        report_structure=report_structure,
        information="\n\n---\n\n".join(state["search_summaries"])
    )

    # 主路径：DeepSeek API + deepseek-reasoner（成稿用推理模型）
    # 想换回本地 Ollama：把下面这行改成 invoke_ollama(model='deepseek-r1:7b', ...)
    # max_tokens=8000：官方文档说默认上限约 4096；但实测该端点会忽略此参数
    # （传 100 仍返回 342 token），这里保留它，只为服务端策略变化时能生效
    result = invoke_deepseek(
        model='deepseek-reasoner',
        system_prompt=answer_prompt,
        user_prompt=f"Generate a research summary using the provided information.",
        max_tokens=8000
    )
    # 去掉思维链，只保留正式回答（API 与 Ollama 两种输出格式都兼容）
    answer = parse_output(result)["response"]
    
    # 方案 B：改用 OpenRouter 调外部模型（GPT-4o / Claude / DeepSeek R1…），取消注释即可切换
    # answer = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=answer_prompt,
    #     user_prompt=f"Generate a research summary using the provided information."
    # )
    
    return {"final_answer": answer}

# 子图：单条子查询的“检索 → 评估 →(条件路由) 摘要 / 网页兜底 → 结束”闭环
# 输入/输出 schema 限定为 QuerySearchStateInput / QuerySearchStateOutput，避免内部字段回写主图
query_search_subgraph = StateGraph(QuerySearchState, input=QuerySearchStateInput, output=QuerySearchStateOutput)

# 注册子图节点（用函数名作为节点名）
query_search_subgraph.add_node(retrieve_rag_documents)
query_search_subgraph.add_node(evaluate_retrieved_documents)
query_search_subgraph.add_node(web_research)
query_search_subgraph.add_node(summarize_query_research)

# 子图入口与流转：检索 → 评估 → 条件路由（摘要 或 网页→摘要）
query_search_subgraph.add_edge(START, "retrieve_rag_documents")
query_search_subgraph.add_edge("retrieve_rag_documents", "evaluate_retrieved_documents")
query_search_subgraph.add_conditional_edges("evaluate_retrieved_documents", route_research)
query_search_subgraph.add_edge("web_research", "summarize_query_research")
query_search_subgraph.add_edge("summarize_query_research", END)

# 主图：拆解 → 分批扇出 → 汇总成稿；config_schema 让 config 里的三个键有类型约束
researcher_graph = StateGraph(ResearcherState, input=ResearcherStateInput, output=ResearcherStateOutput, config_schema=Configuration)

# 注册主图节点（子图以编译产物的形式作为一个节点挂进来）
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(search_queries)
researcher_graph.add_node("search_and_summarize_query", query_search_subgraph.compile())
researcher_graph.add_node(generate_final_answer)

# 主图流转：拆解 → 推进批次 → 条件扇出 → 回来判断是否还有下一批 → 成稿
researcher_graph.add_edge(START, "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "search_queries")
researcher_graph.add_conditional_edges("search_queries", initiate_query_research, ["search_and_summarize_query"])
researcher_graph.add_conditional_edges("search_and_summarize_query", check_more_queries)
researcher_graph.add_edge("generate_final_answer", END)

# 编译成可执行图（模块被 import 时就完成编译）
researcher = researcher_graph.compile()
