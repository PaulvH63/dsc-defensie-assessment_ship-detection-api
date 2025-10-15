import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.model import ShipDetector

# Config from env vars (or defaults)
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n-seg.pt")  # Default to yolo 11 nano seg
CONF = float(os.getenv("CONF", "0.1"))  # Default confidence threshold

# FastAPI app
app = FastAPI(
    title="Ship Detection API",
    version="1.0.0",
    description="Ship Detection API that exposes two endpoints: /image/mask (binary mask) and /image/contour (yellow contours).",
)

# Initialize once
detector = ShipDetector(model_path=MODEL_PATH, conf_threshold=CONF)
class ErrorResponse(BaseModel):
    detail: str

@app.post(
    "/image/mask",
    summary="Return a binary ship mask",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG with binary mask at original size"},
        400: {"model": ErrorResponse, "description": "Invalid image"},
    },
)
async def image_mask(file: UploadFile = File(...)):
    """Return binary mask of detected ships"""
    
    # REad image
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Check if image is valid
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict mask
    mask = detector.predict_mask(image)
    
    # Check if mask is valid
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise HTTPException(status_code=500, detail="Encoding failed")
        
    # Encode and return as PNG
    _, buf = cv2.imencode(".png", mask)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.post(
    "/image/contour",
    summary="Return original image with yellow ship contours",
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG with yellow contours"},
        400: {"model": ErrorResponse, "description": "Invalid image"},
    },
)
async def image_contour(file: UploadFile = File(...)):
    """Return original image with yellow contours"""
    # Read image
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Check if image is valid
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict contours
    contoured = detector.predict_contours(image)

    # Check if contours are valid
    ok, buf = cv2.imencode(".png", contoured)
    if not ok:
        raise HTTPException(status_code=500, detail="Encoding failed")
    contoured_bgr = cv2.cvtColor(contoured, cv2.COLOR_RGB2BGR)
    
    # Encode and return as PNG
    _, buf = cv2.imencode(".png", contoured_bgr)    
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")