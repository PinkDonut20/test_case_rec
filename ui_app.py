import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageDraw

API_URL = os.getenv("OCR_API_URL", "http://localhost:8000/process")
UI_OUT_DIR = Path(os.getenv("UI_OUTPUT_DIR", "ui_outputs"))
UI_OUT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_CSS = """
:root {
  --bg: #0a0f1f;
  --card: #111a33;
  --card-2: #172447;
  --accent: #5eead4;
  --accent-2: #60a5fa;
  --text: #e6eefc;
  --muted: #9fb2d9;
}
.gradio-container {
  background: radial-gradient(1200px 500px at 10% -20%, #1e2f65 0%, transparent 60%),
              radial-gradient(1000px 600px at 90% -10%, #0f766e 0%, transparent 45%),
              var(--bg);
}
.panel {
  background: linear-gradient(160deg, var(--card), var(--card-2));
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 18px;
  padding: 12px;
}
.badge {
  display:inline-block;
  background: rgba(94,234,212,.16);
  color: var(--accent);
  border:1px solid rgba(94,234,212,.35);
  border-radius:999px;
  padding:4px 10px;
  font-size:12px;
}
.hint {
  color: var(--muted);
  font-size: 13px;
}
"""


def _draw_boxes(image: Image.Image, lines: list[dict[str, Any]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for line in lines:
        bbox = line.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        draw.rectangle((x, y, x + w, y + h), outline=(94, 234, 212), width=3)
    return out


def _save_payload(payload: dict[str, Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = UI_OUT_DIR / f"result_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def process_image(image: Image.Image):
    if image is None:
        return None, "", "", None, "", "⚠️ Загрузите изображение документа"

    temp_path = UI_OUT_DIR / "_upload_tmp.jpg"
    image.save(temp_path, format="JPEG")

    try:
        with temp_path.open("rb") as f:
            resp = requests.post(API_URL, files={"file": ("document.jpg", f, "image/jpeg")}, timeout=240)
    except Exception as e:
        return None, "", "", None, "", f"❌ Ошибка подключения к API: {e}"

    if resp.status_code != 200:
        return None, "", "", None, "", f"❌ Ошибка API {resp.status_code}: {resp.text[:300]}"

    payload = resp.json()
    ocr = payload.get("ocr", {})
    lines = ocr.get("lines", [])
    full_text = ocr.get("full_text", "")
    fields = payload.get("fields", {})

    annotated = _draw_boxes(image, lines)
    json_path = _save_payload(payload)

    fields_pretty = json.dumps(fields, ensure_ascii=False, indent=2)
    payload_pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    status = f"✅ Готово: строк OCR = {ocr.get('lines_count', len(lines))}, файл = {Path(json_path).name}"

    return annotated, full_text, fields_pretty, json_path, payload_pretty, status


def clear_all():
    return None, None, "", "", None, "", "🧹 Очищено"


with gr.Blocks(title="Doc OCR Studio", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class='panel'>
          <h1 style='margin:4px 0;color:#e6eefc'>✨ Doc OCR Studio</h1>
          <div class='badge'>FastAPI + OCR + Structured JSON</div>
          <p class='hint'>Загрузите фото документа, нажмите <b>Распознать</b> — получите размеченное изображение и готовый JSON.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["panel"]):
            image_in = gr.Image(label="📷 Документ", type="pil")
            with gr.Row():
                run_btn = gr.Button("🚀 Распознать", variant="primary")
                clear_btn = gr.Button("Очистить")
            status_box = gr.Textbox(label="Статус", interactive=False)

        with gr.Column(scale=1, elem_classes=["panel"]):
            image_out = gr.Image(label="🟩 OCR боксы")

    with gr.Tab("Поля"):
        fields_box = gr.Code(label="Извлечённые поля", language="json")

    with gr.Tab("OCR текст"):
        full_text_box = gr.Textbox(label="Полный OCR текст", lines=14)

    with gr.Tab("Полный JSON"):
        payload_box = gr.Code(label="Ответ API", language="json")
        json_file = gr.File(label="Скачать JSON")

    run_btn.click(
        process_image,
        inputs=[image_in],
        outputs=[image_out, full_text_box, fields_box, json_file, payload_box, status_box],
    )

    clear_btn.click(
        clear_all,
        outputs=[image_in, image_out, full_text_box, fields_box, json_file, payload_box, status_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
