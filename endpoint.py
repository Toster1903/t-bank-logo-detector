import io
import os
import sys
from typing import List, Optional

import gdown
import torch
import torchvision
from torchvision import transforms as T
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

class BoundingBox(BaseModel):
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)

class Detection(BaseModel):
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

app = FastAPI(title="T-Bank Logo Detector", version="1.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "faster_best.pth")
MODEL_GDRIVE_ID = os.environ.get("MODEL_GDRIVE_ID", "19Pi2f3Tfz4kWrM8WCmCKO1eXyhQeX5Zj")
SCORE_THR = float(os.environ.get("SCORE_THR", "0.3"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.ToTensor()

if not os.path.exists(MODEL_PATH):
    print("[INFO] downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}", MODEL_PATH, quiet=False)

def load_model(path: str):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and any(k.startswith("module.") or k in model.state_dict() for k in checkpoint.keys()):
        state = checkpoint
    else:
        raise RuntimeError("Unrecognized checkpoint format")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

try:
    model = load_model(MODEL_PATH)
    print("[INFO] model loaded", file=sys.stderr)
except Exception as e:
    model = None
    print("[ERROR] failed to load model:", str(e), file=sys.stderr)

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Model not loaded", detail="Check server logs").dict())
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
            x_min = max(0, min(x_min, pil_img.width))
            x_max = max(0, min(x_max, pil_img.width))
            y_min = max(0, min(y_min, pil_img.height))
            y_max = max(0, min(y_max, pil_img.height))
            detections.append(Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)))
        return DetectionResponse(detections=detections)
    except Exception as e:
        return JSONResponse(status_code=500, content=ErrorResponse(error="Internal server error", detail=str(e)).dict())

@app.post("/detect_debug")
async def detect_debug(file: UploadFile = File(...)):
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
        draw.text((x1, y2 + 2), f"{score:.2f}")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
