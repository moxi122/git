from fastapi import APIRouter

router = APIRouter(tags=["knowledge"])


@router.get("/health")
async def health():
    """存活探针：不碰 LLM、不碰向量库，保证永远秒回。"""
    return {"status": "ok", "version": "0.1.0"}
