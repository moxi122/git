import os
import re
import shutil
from ollama import chat
from tavily import TavilyClient
from pydantic import BaseModel
from langchain_community.document_loaders import CSVLoader, TextLoader, PDFPlumberLoader
from src.assistant.vector_db import add_documents

# 相关性判定的结构化输出 schema：字段名 is_relevant 会写进 JSON Schema，约束模型只回这一键
class Evaluation(BaseModel):
    is_relevant: bool

# 子查询列表的结构化输出 schema：字段名 queries 会写进 JSON Schema
class Queries(BaseModel):
    queries: list[str]

# 拆分模型输出：本地 Ollama 跑的 R1 会在正文外包一层 <think>...</think> 思维链，
# 而 DeepSeek 官方 API 的 deepseek-reasoner 把思考过程放在 reasoning_content 字段里，
# 正文本身不带标签。所以这里做双模式兼容：
#   有 <think> 标签 -> 拆成 reasoning + response
#   没有标签       -> 整段当 response，reasoning 置空
def parse_output(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    output_match = re.search(r'</think>\s*(.*?)$', text, re.DOTALL)

    if think_match and output_match:
        return {
            "reasoning": think_match.group(1).strip(),
            "response": output_match.group(1).strip()
        }

    return {
        "reasoning": "",
        "response": text.strip()
    }

# 把 Document 列表拍平成带来源标注的纯文本，供 prompt 使用
def format_documents_with_metadata(documents):
    """
    把 Document 列表转换成带元信息的格式化字符串。

    参数：
        documents: Document 对象列表

    返回：
        包含文档正文与来源的字符串
    """
    formatted_docs = []
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown source')
        formatted_doc = f"Source: {source}\nContent: {doc.page_content}"
        formatted_docs.append(formatted_doc)

    return "\n\n---\n\n".join(formatted_docs)

# 方案 A：调用本地 Ollama（本项目默认走这条路，无需 API Key）
def invoke_ollama(model, system_prompt, user_prompt, output_format=None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = chat(
        messages=messages,
        model=model,
        # 传了 schema 就用 Ollama 的 format 参数做 JSON 约束输出
        format=output_format.model_json_schema() if output_format else None
    )

    if output_format:
        return output_format.model_validate_json(response.message.content)
    else:
        return response.message.content
    
def invoke_llm(
    model,  # 指定 OpenRouter 上的模型名，如 gpt-4o-mini
    system_prompt,
    user_prompt,
    output_format=None,
    temperature=0
):
        
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model, 
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base= "https://openrouter.ai/api/v1",
    )
    
    # 传了 output_format 就走 LangChain 的结构化输出（用 Pydantic 约束 JSON）
    if output_format:
        llm = llm.with_structured_output(output_format)
    
    # 调用模型
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.invoke(messages)
    
    if output_format:
        return response
    return response.content # 纯文本回答

# 方案 C：调用 DeepSeek 官方 API（OpenAI 兼容端点），Key 从 .env 的 DEEPSEEK_API_KEY 读取
# 注意模型选型：结构化输出节点必须用 deepseek-chat，deepseek-reasoner 不支持
# response_format / 函数调用，做不了 JSON 约束；自由文本节点可以用 reasoner
def invoke_deepseek(model, system_prompt, user_prompt, output_format=None, temperature=0, max_tokens=None):
    """
    用 DeepSeek 官方 API 调用模型。

    参数：
        model: deepseek-chat（结构化输出）或 deepseek-reasoner（自由文本）
        output_format: Pydantic 模型，传了就走结构化输出
        max_tokens: 输出上限；不传则用服务端默认。实测 DeepSeek 官方端点会忽略该参数
    """
    from langchain_openai import ChatOpenAI

    llm_kwargs = {
        "model": model,
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        # 默认官方地址；换成第三方网关/内网代理时改 .env 里的 DEEPSEEK_BASE_URL 即可
        # 用 or 兜底，避免 .env 里写成空字符串时把 base_url 变成 ""
        "base_url": os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
    }
    # reasoner 不支持 temperature 等采样参数，只给 chat 系列传
    if "reasoner" not in model:
        llm_kwargs["temperature"] = temperature
    if max_tokens:
        llm_kwargs["max_tokens"] = max_tokens

    llm = ChatOpenAI(**llm_kwargs)

    # 传了 output_format 就走结构化输出；这里必须显式指定 method="function_calling"：
    # langchain 默认用 response_format={"type": "json_schema"}，DeepSeek 会直接回
    # 400 "This response_format type is unavailable now"，换成 tools 传 schema 才能跑通
    if output_format:
        llm = llm.with_structured_output(output_format, method="function_calling")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.invoke(messages)

    if output_format:
        return response
    return response.content

def tavily_search(query, include_raw_content=True, max_results=3):
    """ 用 Tavily API 做网页检索。

    参数：
        query (str): 要执行的检索词
        include_raw_content (bool): 是否把网页正文 raw_content 一并返回
        max_results (int): 最多返回多少条结果

    返回：
        dict: 检索响应，其中 results 是结果列表，每项包含：
            - title (str): 结果标题
            - url (str): 结果链接
            - content (str): 内容摘要
            - raw_content (str): 可用时返回网页全文"""

    tavily_client = TavilyClient()
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content
    )

def get_report_structures(reports_folder="report_structures"):
    """
    从指定目录下的 .md / .txt 文件读取报告结构。
    文件名形如 report_name.md 或 report_name.txt，文件名即结构名。
    返回：报告结构字典。
    """
    report_structures = {}

    # 目录不存在就创建，避免后续 listdir 抛异常
    os.makedirs(reports_folder, exist_ok=True)

    try:
        # 遍历目录下所有 .md / .txt 文件
        for filename in os.listdir(reports_folder):
            if filename.endswith(('.md', '.txt')):
                report_name = os.path.splitext(filename)[0]  # 去掉扩展名，作为下拉框显示的结构名
                file_path = os.path.join(reports_folder, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        report_structures[report_name] = {
                            "content": content
                        }
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

    except Exception as e:
        print(f"Error accessing reports folder: {str(e)}")

    return report_structures

def process_uploaded_files(uploaded_files):
    temp_folder = "temp_files"
    os.makedirs(temp_folder, exist_ok=True)

    try:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            temp_file_path = os.path.join(temp_folder, uploaded_file.name)

            # 先落盘到临时目录：loader 需要真实文件路径
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # 按扩展名选择对应的 loader
            if file_extension == "csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension in ["txt", "md"]:
                loader = TextLoader(temp_file_path)
            elif file_extension == "pdf":
                loader = PDFPlumberLoader(temp_file_path)
            else:
                continue

            # 读取 → 切分 → 写入向量库
            docs = loader.load()
            add_documents(docs)

        return True
    finally:
        # 无论成功失败都清掉临时目录
        shutil.rmtree(temp_folder, ignore_errors=True)
