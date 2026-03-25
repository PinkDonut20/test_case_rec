import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import requests
from PIL import Image, ImageDraw

API_URL = os.getenv("OCR_API_URL", "http://localhost:8000/process")
UI_OUT_DIR = Path(os.getenv("UI_OUTPUT_DIR", "ui_outputs"))
UI_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_boxes(image: Image.Image, lines: list[dict[str, Any]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for line in lines:
        bbox = line.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=3)
    return out


def process_image(image: Image.Image):
    if image is None:
        return None, "", "", None, "Загрузите изображение"

    temp_path = UI_OUT_DIR / "_upload_tmp.jpg"
    image.save(temp_path, format="JPEG")

    with temp_path.open("rb") as f:
        resp = requests.post(API_URL, files={"file": ("document.jpg", f, "image/jpeg")}, timeout=180)

    if resp.status_code != 200:
        return None, "", "", None, f"Ошибка API: {resp.status_code} {resp.text[:500]}"

    payload = resp.json()
    ocr_lines = payload.get("ocr", {}).get("lines", [])
    full_text = payload.get("ocr", {}).get("full_text", "")
    fields = payload.get("fields", {})

    annotated = _draw_boxes(image, ocr_lines)

    out_path = UI_OUT_DIR / "result.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields_json = json.dumps(fields, ensure_ascii=False, indent=2)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return annotated, full_text, fields_json, str(out_path), payload_json


with gr.Blocks(title="Document OCR Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# OCR документов
Загрузите фото документа, нажмите **Распознать**.
Приложение отправит изображение в API (`/process`), покажет боксы OCR и сохранит JSON.
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Изображение документа", type="pil")
            run_btn = gr.Button("Распознать", variant="primary")
        with gr.Column(scale=1):
            image_out = gr.Image(label="Результат OCR (боксы)")

    full_text_box = gr.Textbox(label="Полный OCR-текст", lines=10)
    fields_box = gr.Code(label="Извлечённые поля", language="json")

    with gr.Row():
        json_file = gr.File(label="Сохранённый JSON")
    payload_box = gr.Code(label="Полный ответ API", language="json")

    run_btn.click(
        process_image,
        inputs=[image_in],
        outputs=[image_out, full_text_box, fields_box, json_file, payload_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
