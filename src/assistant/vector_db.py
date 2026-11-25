import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

VECTOR_DB_PATH = "database"
 
def get_or_create_vector_db():
    """获取向量库：已有库直接加载，没有就现场建库。"""
    embeddings = HuggingFaceEmbeddings()

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # 库已存在（目录非空）：直接加载，不再重复做 embedding
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    else:
        # 库不存在：读取 ./files 目录并现场建库
        loader = DirectoryLoader("./files")
        docs = loader.load()

        # 语义切分：按语义断点切块，而不是按固定长度硬切
        semantic_text_splitter = SemanticChunker(embeddings)
        documents = semantic_text_splitter.split_documents(docs)

        # SemanticChunker 没有最大块长参数，切出来的块可能过长，
        # 所以再用 RecursiveCharacterTextSplitter 兜底压到 2000 字符、重叠 400
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        split_documents = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(split_documents, embeddings, persist_directory=VECTOR_DB_PATH)

    return vectorstore

def add_documents(documents):
    """
    把新文档追加进已有的向量库。

    参数：
        documents: 待写入向量库的文档列表
    """
    embeddings = HuggingFaceEmbeddings()
    
    # 语义切分：按语义断点切块
    semantic_text_splitter = SemanticChunker(embeddings)
    documents = semantic_text_splitter.split_documents(documents)

    # SemanticChunker 没有最大块长参数，切出来的块可能过长，
    # 所以再用 RecursiveCharacterTextSplitter 兜底压到 2000 字符、重叠 400
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    split_documents = text_splitter.split_documents(documents)

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        # 库已存在：增量追加
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        vectorstore.add_documents(split_documents)
    else:
        # 库不存在：新建
        vectorstore = Chroma.from_documents(
            split_documents,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    return vectorstore
