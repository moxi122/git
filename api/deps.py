"""依赖注入层：把"真实实现"与"路由代码"隔开。

好处有三：
1. 路由里不出现 torch / chromadb / LLM，导入快，单测不需要真模型；
2. 测试可以用 app.dependency_overrides 换成 fake；
3. 真实实现是惰性导入，服务启动不会因为缺 API key 直接崩。
"""


def get_graph():
    from src.assistant.graph import researcher

    return researcher


def get_retriever():
    from src.assistant.retrieval import hybrid_retrieve

    def retrieve(query: str, top_k: int):
        return hybrid_retrieve(query, top_n=top_k)

    return retrieve


def get_ingestor():
    from src.assistant.utils import process_uploaded_files

    return process_uploaded_files
