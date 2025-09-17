# endpoint.py
import io
import os
import sys
import tempfile
import requests
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision import transforms as T

# ---------------- Pydantic контракт (строго) ----------------
class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

# ---------------- FastAPI init ----------------
app = FastAPI(title="T-Bank Logo Detector", version="1.0")

# ---------------- Configs через env ----------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/dmitriy/vs_code/T-bank/faster_best.pth")
MODEL_URL = os.environ.get("MODEL_URL")  # если указан, контейнер попытается скачать
SCORE_THR = float(os.environ.get("SCORE_THR", "0.3"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.ToTensor()

# ---------------- Helpers ----------------
def download_model(url: str, target_path: str):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def load_model(path: str, url: Optional[str] = None):
    # если нет локального файла и есть URL — скачиваем
    if not os.path.exists(path):
        if url:
            print(f"[INFO] model not found at {path}, downloading from MODEL_URL", file=sys.stderr)
            download_model(url, path)
        else:
            raise FileNotFoundError(f"Model file not found at {path} and MODEL_URL not provided")
    # модельная архитектура: Faster R-CNN resnet50 FPN, num_classes=2
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    checkpoint = torch.load(path, map_location=device)
    # поддержка двух вариантов сохранения: либо dict with key "model_state", либо state_dict directly
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and any(k.startswith("module.") or k in model.state_dict() for k in checkpoint.keys()):
        # возможно это state_dict
        state = checkpoint
    else:
        raise RuntimeError("Unrecognized checkpoint format")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ---------------- Load model on startup ----------------
try:
    MODEL_PATH_FINAL = MODEL_PATH
    if MODEL_URL and not os.path.exists(MODEL_PATH_FINAL):
        # download to MODEL_PATH
        os.makedirs(os.path.dirname(MODEL_PATH_FINAL), exist_ok=True)
        download_model(MODEL_URL, MODEL_PATH_FINAL)
    model = load_model(MODEL_PATH_FINAL, url=None)
    print("[INFO] model loaded", file=sys.stderr)
except Exception as e:
    model = None
    print("[ERROR] failed to load model:", str(e), file=sys.stderr)

# ---------------- Core /detect endpoint ----------------
@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении
    """
    if model is None:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Model not loaded", detail="Check server logs").dict())

    # проверка формата
    filename = file.filename or ""
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return JSONResponse(status_code=400, content=ErrorResponse(error="Unsupported file format", detail="Use JPEG, PNG, BMP or WEBP").dict())

    try:
        data = await file.read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        img_tensor = transform(pil_img).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        detections = []
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if float(score) < SCORE_THR:
                continue
            x_min, y_min, x_max, y_max = [int(v) for v in box.tolist()]
            # clip to image bounds
            x_min = max(0, min(x_min, pil_img.width))
            x_max = max(0, min(x_max, pil_img.width))
            y_min = max(0, min(y_min, pil_img.height))
            y_max = max(0, min(y_max, pil_img.height))
            detections.append(Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)))

        return DetectionResponse(detections=detections)

    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Internal server error", detail=str(e)).dict())

# ---------------- Optional: debug endpoint returning PNG image with boxes ----------------
@app.post("/detect_debug")
async def detect_debug(file: UploadFile = File(...)):
    """
    Debug endpoint: вернёт PNG с нарисованными боксами (GT нет).
    Нужен только для локальной отладки.
    """
    if model is None:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Model not loaded", detail="Check server logs").dict())

    data = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = transform(pil_img).to(device)

    with torch.no_grad():
        outputs = model([tensor])[0]

    draw = ImageDraw.Draw(pil_img)
    for box, score in zip(outputs["boxes"], outputs["scores"]):
        if float(score) < SCORE_THR:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y2+2), f"{score:.2f}")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
