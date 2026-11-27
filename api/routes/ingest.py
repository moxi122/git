import asyncio
import os
from dataclasses import dataclass

from fastapi import APIRouter, Depends, File, UploadFile

from api.config import max_upload_bytes
from api.deps import get_ingestor
from api.errors import AppError
from api.schemas import IngestResponse

router = APIRouter(tags=["knowledge"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}
CHUNK_SIZE = 1024 * 1024


@dataclass
class UploadPayload:
    """Streamlit 的 UploadedFile 鸭子类型：utils 只用到 .name 和 .getvalue()。"""

    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    ingestor=Depends(get_ingestor),
):
    # 只取 basename，防止 ../../etc 之类的路径穿越
    filename = os.path.basename(file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AppError(
            "UNSUPPORTED_FILE",
            f"仅支持 {sorted(ALLOWED_EXTENSIONS)}，收到 {ext or '无扩展名'}",
            400,
        )

    # 分块读，边读边计数，避免 100MB 文件先落盘再被拒
    limit = max_upload_bytes()
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise AppError("FILE_TOO_LARGE", f"单文件上限 {limit} 字节", 413)
        chunks.append(chunk)

    data = b"".join(chunks)
    if not data:
        raise AppError("EMPTY_FILE", "上传文件为空", 400)

    try:
        ok = await asyncio.to_thread(ingestor, [UploadPayload(name=filename, data=data)])
    except Exception as exc:
        raise AppError("INGEST_FAILED", f"入库失败: {exc}", 502)

    return IngestResponse(filename=filename, ingested=bool(ok))
