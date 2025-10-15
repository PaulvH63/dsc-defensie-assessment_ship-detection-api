import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from src.model import ShipDetector

# Config from env vars (or defaults)
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n-seg.pt")  # Default to yolo 11 nano seg
CONF = float(os.getenv("CONF", "0.1"))  # Default confidence threshold

# FastAPI app
app = FastAPI(
    title="Ship Detection API",
    version="1.0.0",
    description="Ship detector with two endpoints: /image/mask and /image/contour.",
)

# Initialize once
detector = ShipDetector(model_path=MODEL_PATH, conf_threshold=CONF)

@app.post("/image/mask")
async def image_mask(file: UploadFile = File(...)):
    """Return binary mask of detected ships"""
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict mask
    mask = detector.predict_mask(image)

    # Encode and return as PNG
    _, buf = cv2.imencode(".png", mask)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.post("/image/contour")
async def image_contour(file: UploadFile = File(...)):
    """Return original image with yellow contours"""
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict contours
    contoured = detector.predict_contours(image)

    # Encode and return as PNG
    contoured_bgr = cv2.cvtColor(contoured, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", contoured_bgr)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")