from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

app = FastAPI(title="Prescription OCR with TrOCR")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for TrOCR model
trocr_processor = None
trocr_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_trocr_model():
    """Load TrOCR model for handwritten text recognition"""
    global trocr_processor, trocr_model
    
    if trocr_model is not None:
        return
    
    try:
        print(f"Loading TrOCR model on {device}...")
        model_name = "microsoft/trocr-base-handwritten"
        
        trocr_processor = AutoProcessor.from_pretrained(model_name)
        trocr_model = AutoModelForVision2Seq.from_pretrained(model_name)
        trocr_model.to(device)
        trocr_model.eval()
        
        print("TrOCR model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading TrOCR model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_trocr_model()
    except Exception as e:
        print(f"Warning: Could not load TrOCR model on startup: {e}")
        print("The server will start but OCR will not work until the model is loaded.")

@app.get("/")
async def root():
    return {"message": "Prescription OCR API with TrOCR", "status": "running"}

@app.get("/health")
async def health_check():
    """Check if model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": trocr_model is not None,
        "device": device
    }

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """Convert uploaded image to PIL Image"""
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def perform_ocr(image: Image.Image) -> str:
    """
    Use TrOCR to extract text from the image
    """
    global trocr_processor, trocr_model
    
    if trocr_model is None:
        raise HTTPException(status_code=503, detail="TrOCR model not loaded")
    
    try:
        # Process image
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        
        # Decode the generated text
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
        
    except Exception as e:
        print(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    """
    Upload an image and get OCR results
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        # Perform OCR
        extracted_text = perform_ocr(image)
        
        # Create a copy of the image for display
        image_array = np.array(image)
        
        # Convert back to base64 for frontend
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "extracted_text": extracted_text,
            "image_preview": f"data:image/png;base64,{image_base64}",
            "text_length": len(extracted_text),
            "model": "microsoft/trocr-base-handwritten"
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



