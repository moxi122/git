import pyperclip
import streamlit as st
import streamlit_nested_layout
from src.assistant.graph import researcher
from src.assistant.utils import get_report_structures, process_uploaded_files
from dotenv import load_dotenv

load_dotenv()

def generate_response(user_input, enable_web_search, report_structure, max_search_queries):
    """
    调用 researcher 智能体生成回答，并把每一步的中间结果流式渲染出来。
    """
    # 初始化 researcher 的输入状态：入口只需要 user_instructions，其余字段由各节点写入
    initial_state = {
        "user_instructions": user_input,
    }
    
    # LangGraph 运行时配置：configurable 里的三个键会被各节点的 config 参数读取
    config = {"configurable": {
        "enable_web_search": enable_web_search,
        "report_structure": report_structure,
        "max_search_queries": max_search_queries,
    }}

    # 创建全局“Researcher 运行中”的状态容器，下面三个 expander 都挂在它里面
    langgraph_status = st.status("**Researcher Running...**", state="running")

    # 先创建三个 expander 以固定展示顺序（此时为空，等流式输出到来再填充）
    with langgraph_status:
        generate_queries_expander = st.expander("Generate Research Queries", expanded=False)
        search_queries_expander = st.expander("Search Queries", expanded=True)
        final_answer_expander = st.expander("Generate Final Answer", expanded=False)

        steps = []

        # 运行 researcher 图并流式读取输出：每一轮只包含一个节点产生的增量
        for output in researcher.stream(initial_state, config=config):
            for key, value in output.items():
                expander_label = key.replace("_", " ").title()

                if key == "generate_research_queries":
                    with generate_queries_expander:
                        st.write(value)

                elif key.startswith("search_and_summarize_query"):
                    with search_queries_expander:
                        with st.expander(expander_label, expanded=False):
                            st.write(value)

                elif key == "generate_final_answer":
                    with final_answer_expander:
                        st.write(value)

                steps.append({"step": key, "content": value})

    # 全部跑完，把整体状态标记为完成
    langgraph_status.update(state="complete", label="**Using Langgraph** (Research completed)")

    # 返回最后一步（generate_final_answer 节点）的输出
    return steps[-1]["content"] if steps else "No response generated"

def clear_chat():
    st.session_state.messages = []
    st.session_state.processing_complete = False
    st.session_state.uploader_key = 0

def main():
    st.set_page_config(page_title="DeepSeek RAG Researcher", layout="wide")

    # 初始化 Streamlit 会话状态：首次进入时创建，之后每次 rerun 都复用
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_report_structure" not in st.session_state:
        st.session_state.selected_report_structure = None
    if "max_search_queries" not in st.session_state:
        st.session_state.max_search_queries = 5  # 默认 5 条子查询
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False  # 标记“已上传但还没入库”

    # 标题行 + 清空按钮
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("📄 RAG Researcher with DeepSeek R1")
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()

    # 侧边栏配置区
    st.sidebar.title("Research Settings")

    # 侧边栏：报告结构选择器，选项来自 report_structures/ 目录下的 .md/.txt 文件
    report_structures = get_report_structures()
    default_report = "standard report"

    selected_structure = st.sidebar.selectbox(
        "Select Report Structure",
        options=list(report_structures.keys()),
        index=list(map(str.lower, report_structures.keys())).index(default_report)
    )

    st.session_state.selected_report_structure = report_structures[selected_structure]

    # 子查询数量上限
    st.session_state.max_search_queries = st.sidebar.number_input(
        "Max Number of Search Queries",
        min_value=1,
        max_value=10,
        value=st.session_state.max_search_queries,
        help="Set the maximum number of search queries to be made. (1-10)"
    )
    
    enable_web_search = st.sidebar.checkbox("Enable Web Search", value=False)

    # 文件上传：新文档入库后即可被本地检索使用
    uploaded_files = st.sidebar.file_uploader(
        "Upload New Documents",
        type=["pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # 有文件传上来但还没入库：置位“就绪”标记，用于显示处理按钮
    if uploaded_files:
        st.session_state.files_ready = True  # 标记文件已就绪
        st.session_state.processing_complete = False  # 复位处理状态

    # 仅在“已上传、未处理”时显示 Process Files 按钮
    if st.session_state.files_ready and not st.session_state.processing_complete:
        process_button_placeholder = st.sidebar.empty()  # 占位符，方便按钮原地被替换成进度条

        with process_button_placeholder.container():
            process_clicked = st.button("Process Files", use_container_width=True)

        if process_clicked:
            with process_button_placeholder:
                with st.status("Processing files...", expanded=False) as status:
                    # 真正的入库动作：切分 + 写入 Chroma（实现见 utils.process_uploaded_files）
                    if process_uploaded_files(uploaded_files):
                        st.session_state.processing_complete = True
                        st.session_state.files_ready = False  # 复位就绪标记
                        st.session_state.uploader_key += 1  # 更换 key，重置上传组件以便继续上传

                    status.update(label="Files processed successfully!", state="complete", expanded=False)
                    # st.rerun()

    # 渲染历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])  # 正常显示消息正文

            # 只有助手消息在下方带复制按钮
            if message["role"] == "assistant":
                if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                    pyperclip.copy(message["content"])

    # 输入框 + 回答处理
    if user_input := st.chat_input("Type your message here..."):
        # 先把用户消息写入会话
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 生成并展示助手回答
        report_structure = st.session_state.selected_report_structure["content"]
        assistant_response = generate_response(
            user_input, 
            enable_web_search, 
            report_structure,
            st.session_state.max_search_queries
        )

        # 把助手消息存入会话
        st.session_state.messages.append({"role": "assistant", "content": assistant_response["final_answer"]})

        with st.chat_message("assistant"):
            st.write(assistant_response["final_answer"])  # 助手回答正文

            # 助手消息下方的复制按钮
            if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                pyperclip.copy(assistant_response["final_answer"])

if __name__ == "__main__":
    main()
