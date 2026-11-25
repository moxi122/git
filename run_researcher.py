from src.assistant.graph import researcher
from src.assistant.vector_db import get_or_create_vector_db
from dotenv import load_dotenv

load_dotenv()

report_structure = """
1. Introduction
- Brief overview of the research topic or question.
- Purpose and scope of the report.

2. Main Body
- For each section (e.g., Section 1, Section 2, Section 3, etc.):
  - Provide a subheading related to the key aspect of the research.
  - Include explanation, findings, and supporting details.

3. Key Takeaways
- Bullet points summarizing the most important insights or findings.

4. Conclusion
- Final summary of the research.
- Implications or relevance of the findings.
"""

# 定义初始状态：命令行脚本只提供用户指令，其余字段交给节点填充
initial_state = {
    "user_instructions": "Can you help me understand the current state of AI reasoning models, particularly DeepSeek R-1? I keep hearing about breakthroughs in mathematical reasoning and problem-solving capabilities, but I want to know how these models are actually being implemented and used in real applications. I am Really interested in learning about its performance compared to other LLMs, what makes its training approach unique, and if there are any concerns about reliability or limitations. Looking for recent benchmarks and real-world applications, not just theoretical capabilities.",
}

# LangGraph 运行时配置：关闭网页检索 + 5 条子查询 + 上面这份报告结构
config = {
  "configurable": {
    "enable_web_search": False,
    "report_structure": report_structure,
    "max_search_queries": 5
}}

# 初始化向量库：首次运行会读取 ./files 现场建库
# 注意：运行前必须先把自有文档放进 /files 目录，目录为空这里会直接报错
vector_db = get_or_create_vector_db()

# 运行研究者图，逐个节点打印输出（便于观察每一步的增量）
for output in researcher.stream(initial_state, config=config):
    for key, value in output.items():
        print(f"Finished running: **{key}**")
        print(value)
