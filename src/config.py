import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    output_dir: Path
    extractor_mode: str
    ocr_backend: str
    use_gpu: bool
    ocr_conf_threshold: float
    ocr_min_box_area: int
    hf_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    lighton_model: str



def load_settings() -> Settings:
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        output_dir=output_dir,
        extractor_mode=os.getenv("EXTRACTOR_MODE", "hf").lower(),
        ocr_backend=os.getenv("OCR_BACKEND", "lighton").lower(),
        use_gpu=os.getenv("USE_GPU", "0") == "1",
        ocr_conf_threshold=float(os.getenv("OCR_CONF_THRESHOLD", "0.20")),
        ocr_min_box_area=int(os.getenv("OCR_MIN_BOX_AREA", "60")),
        hf_model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        lighton_model=os.getenv("LIGHTON_MODEL", "lightonai/LightOnOCR-2-1B"),
    )
