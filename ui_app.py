import base64
import io
import json
import os
from pathlib import Path

import requests
from flask import Flask, Response, render_template_string, request
from PIL import Image, ImageDraw

API_URL = os.getenv("OCR_API_URL", "http://localhost:8000/process")
UI_OUT_DIR = Path(os.getenv("UI_OUTPUT_DIR", "ui_outputs"))
UI_OUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Document OCR UI</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 2rem; background:#0f172a; color:#e2e8f0; }
    .card { background:#111827; border:1px solid #334155; border-radius:14px; padding:1rem 1.2rem; margin-bottom:1rem; }
    .btn { background:#2563eb; color:white; border:0; border-radius:10px; padding:0.6rem 1rem; cursor:pointer; }
    .muted { color:#94a3b8; }
    pre { background:#020617; border:1px solid #334155; border-radius:10px; padding:1rem; overflow:auto; }
    input[type=file] { margin-bottom: 0.8rem; }
    a { color:#60a5fa; }
    img { max-width: 100%; border-radius: 10px; border:1px solid #334155; }
  </style>
</head>
<body>
  <h1>OCR документов</h1>
  <p class="muted">Загрузите изображение и нажмите «Распознать». UI отправит файл в API и сохранит JSON локально.</p>

  <div class="card">
    <form method="post" action="/process" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required /><br>
      <button class="btn" type="submit">Распознать</button>
    </form>
  </div>

  {% if error %}
  <div class="card">
    <h3>Ошибка</h3>
    <pre>{{ error }}</pre>
  </div>
  {% endif %}

  {% if annotated_b64 %}
  <div class="card">
    <h3>Разметка OCR (боксы)</h3>
    <img src="data:image/jpeg;base64,{{ annotated_b64 }}" alt="OCR boxes" />
  </div>
  {% endif %}

  {% if result %}
  <div class="card">
    <h3>Извлечённые поля</h3>
    <pre>{{ fields }}</pre>
  </div>

  <div class="card">
    <h3>Полный OCR текст</h3>
    <pre>{{ full_text }}</pre>
  </div>

  <div class="card">
    <h3>Полный JSON</h3>
    <pre>{{ result }}</pre>
    <p>Сохранено в: <code>{{ json_path }}</code> | <a href="/download-json">Скачать JSON</a></p>
  </div>
  {% endif %}
</body>
</html>
"""


def _draw_boxes(image_bytes: bytes, lines: list[dict]) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for line in lines:
        bbox = line.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = [int(v) for v in bbox]
        draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/")
def index():
    return render_template_string(HTML, result=None, fields=None, full_text=None, json_path=None, error=None, annotated_b64=None)


@app.post("/process")
def process():
    uploaded = request.files.get("file")
    if uploaded is None:
        return render_template_string(HTML, result=None, fields=None, full_text=None, json_path=None, error="Файл не передан", annotated_b64=None)

    image_bytes = uploaded.read()
    files = {"file": (uploaded.filename or "document.jpg", io.BytesIO(image_bytes), uploaded.mimetype or "image/jpeg")}

    try:
        resp = requests.post(API_URL, files=files, timeout=240)
    except Exception as e:
        return render_template_string(HTML, result=None, fields=None, full_text=None, json_path=None, error=f"Ошибка запроса к API: {e}", annotated_b64=None)

    if resp.status_code != 200:
        return render_template_string(
            HTML,
            result=None,
            fields=None,
            full_text=None,
            json_path=None,
            error=f"API вернул {resp.status_code}: {resp.text[:1000]}",
            annotated_b64=None,
        )

    payload = resp.json()
    lines = payload.get("ocr", {}).get("lines", [])
    annotated_b64 = _draw_boxes(image_bytes, lines)

    out_path = UI_OUT_DIR / "result.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = json.dumps(payload.get("fields", {}), ensure_ascii=False, indent=2)
    full_text = payload.get("ocr", {}).get("full_text", "")
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)

    return render_template_string(
        HTML,
        result=pretty,
        fields=fields,
        full_text=full_text,
        json_path=str(out_path),
        error=None,
        annotated_b64=annotated_b64,
    )


@app.get("/download-json")
def download_json():
    out_path = UI_OUT_DIR / "result.json"
    if not out_path.exists():
        return Response("result.json пока не создан", status=404)
    return Response(out_path.read_bytes(), mimetype="application/json", headers={"Content-Disposition": "attachment; filename=result.json"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
