import asyncio

from fastapi import APIRouter, Depends

from api.deps import get_retriever
from api.errors import AppError
from api.schemas import DocumentOut, RetrieveRequest, RetrieveResponse

router = APIRouter(tags=["research"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest, retriever=Depends(get_retriever)):
    try:
        docs = await asyncio.to_thread(retriever, request.query, request.top_k)
    except Exception as exc:
        raise AppError("RETRIEVE_FAILED", f"检索失败: {exc}", 502)

    return RetrieveResponse(
        query=request.query,
        top_k=request.top_k,
        documents=[
            DocumentOut(
                content=doc.page_content,
                source=str(doc.metadata.get("source", "unknown")),
                score=doc.metadata.get("score"),
            )
            for doc in docs
        ],
    )
