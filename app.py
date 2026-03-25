from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.config import load_settings
from src.pipeline import DocumentPipeline

settings = load_settings()
pipeline = DocumentPipeline(settings)

app = FastAPI(title="doc-ocr-llm", version="6.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "extractor_mode": settings.extractor_mode,
        "ocr_backend": settings.ocr_backend,
        "use_gpu": settings.use_gpu,
        "lighton_model": settings.lighton_model if settings.ocr_backend == "lighton" else None,
        "hf_model": settings.hf_model if settings.extractor_mode in {"hf", "hybrid"} else None,
        "llm_model": settings.llm_model if settings.extractor_mode in {"api", "hybrid"} else None,
        "llm_base_url": settings.llm_base_url if settings.extractor_mode in {"api", "hybrid"} else None,
    }


@app.post("/process")
async def process(file: UploadFile = File(...)) -> JSONResponse:
    payload = await pipeline.process_upload(file)
    return JSONResponse(payload)
