from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

# Try to import SAM2 (optional - will work without it)
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Will use simple region detection.")

app = FastAPI(title="Prescription OCR with Segmentation + TrOCR")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
trocr_processor = None
trocr_model = None
sam2_model = None
sam2_mask_generator = None
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

def load_sam2_model():
    """Load SAM2 model for segmentation (optional)"""
    global sam2_model, sam2_mask_generator
    
    if not SAM2_AVAILABLE:
        return False
    
    if sam2_model is not None:
        return True
    
    try:
        model_cfg = "sam2_hiera_large.yaml"
        checkpoint_path = "checkpoints/sam2_hiera_large.pt"
        
        if not os.path.exists(checkpoint_path):
            print("SAM2 checkpoint not found. Using simple region detection instead.")
            return False
        
        print(f"Loading SAM2 model on {device}...")
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        print("SAM2 model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"SAM2 not available: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_trocr_model()
    except Exception as e:
        print(f"Warning: Could not load TrOCR model: {e}")
    
    # Try to load SAM2 (optional)
    load_sam2_model()

@app.get("/")
async def root():
    return {
        "message": "Prescription OCR API with Segmentation + TrOCR",
        "status": "running",
        "trocr_loaded": trocr_model is not None,
        "sam2_loaded": sam2_model is not None
    }

@app.get("/health")
async def health_check():
    """Check if models are loaded"""
    return {
        "status": "healthy",
        "trocr_loaded": trocr_model is not None,
        "sam2_loaded": sam2_model is not None,
        "device": device
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def segment_image_sam2(image: np.ndarray) -> List[Dict]:
    """Use SAM2 to segment the image into regions"""
    global sam2_mask_generator
    
    if sam2_mask_generator is None:
        return []
    
    try:
        masks = sam2_mask_generator.generate(image)
        if not masks:
            return []
        
        # Sort by area and return top regions
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        return masks_sorted[:15]  # Top 15 regions
        
    except Exception as e:
        print(f"Error in SAM2 segmentation: {e}")
        return []

def segment_image_simple(image: np.ndarray) -> List[Dict]:
    """Simple region detection using image processing (fallback when SAM2 not available)"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small regions
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Create mask
                mask = np.zeros((h, w), dtype=bool)
                cv2.fillPoly(mask, [contour], True)
                
                regions.append({
                    'segmentation': mask,
                    'area': area,
                    'bbox': [x, y, w_rect, h_rect]
                })
        
        # Sort by area
        regions.sort(key=lambda x: x['area'], reverse=True)
        return regions[:10]  # Top 10 regions
        
    except Exception as e:
        print(f"Error in simple segmentation: {e}")
        return []

def classify_region_handwritten(image: np.ndarray, mask: np.ndarray) -> bool:
    """
    Classify if a region contains handwritten text.
    Uses heuristics: handwritten text has more variation and irregular edges.
    """
    try:
        # Extract the masked region
        masked_region = image[mask]
        
        if len(masked_region) == 0:
            return False
        
        # Convert to grayscale for analysis
        if len(masked_region.shape) == 3:
            gray_region = cv2.cvtColor(masked_region.reshape(-1, masked_region.shape[-1]), cv2.COLOR_RGB2GRAY)
        else:
            gray_region = masked_region.flatten()
        
        # Calculate statistics
        variance = np.var(gray_region)
        mean = np.mean(gray_region)
        
        # Calculate edge density
        mask_2d = mask.reshape(image.shape[:2])
        edges = cv2.Canny((mask_2d * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / np.sum(mask) if np.sum(mask) > 0 else 0
        
        # Heuristics for handwritten text
        # Handwritten text typically has:
        # - Higher variance (irregular strokes)
        # - Higher edge density (more edges)
        # - Less uniform spacing
        
        is_handwritten = (
            variance > 1500 and  # More variation
            edge_density > 0.08 and  # More edges
            mean < 200  # Not too bright (text is usually dark)
        )
        
        return is_handwritten
        
    except Exception as e:
        print(f"Error classifying region: {e}")
        return False

def extract_text_from_region(image: Image.Image, mask: np.ndarray) -> str:
    """Extract text from a specific region using TrOCR"""
    global trocr_processor, trocr_model
    
    if trocr_model is None:
        return ""
    
    try:
        # Extract the region
        image_array = np.array(image)
        masked_image = image_array.copy()
        masked_image[~mask] = 255  # Set non-masked areas to white
        
        # Convert back to PIL Image
        region_image = Image.fromarray(masked_image)
        
        # Crop to bounding box for better results
        mask_2d = mask.reshape(image_array.shape[:2])
        rows = np.any(mask_2d, axis=1)
        cols = np.any(mask_2d, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add padding
            padding = 10
            y_min = max(0, y_min - padding)
            y_max = min(image_array.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image_array.shape[1], x_max + padding)
            
            region_image = region_image.crop((x_min, y_min, x_max, y_max))
        
        # Process with TrOCR
        pixel_values = trocr_processor(images=region_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error extracting text from region: {e}")
        return ""

@app.post("/api/segment")
async def segment_and_ocr(file: UploadFile = File(...)):
    """
    Upload an image, segment it, identify handwritten regions, and extract text from them
    """
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        image_pil = Image.fromarray(image_array)
        
        # Step 1: Segment the image
        if sam2_mask_generator is not None:
            print("Using SAM2 for segmentation...")
            regions = segment_image_sam2(image_array)
        else:
            print("Using simple region detection...")
            regions = segment_image_simple(image_array)
        
        if not regions:
            # Fallback: process entire image
            print("No regions found, processing entire image...")
            full_mask = np.ones((image_array.shape[0], image_array.shape[1]), dtype=bool)
            text = extract_text_from_region(image_pil, full_mask)
            
            return JSONResponse({
                "success": True,
                "extracted_text": text,
                "regions": [],
                "handwritten_regions": 0,
                "total_regions": 0,
                "method": "full_image"
            })
        
        # Step 2: Classify regions and extract text from handwritten ones
        handwritten_texts = []
        region_results = []
        overlay = image_array.copy()
        
        for i, region in enumerate(regions):
            mask = region['segmentation']
            is_handwritten = classify_region_handwritten(image_array, mask)
            
            if is_handwritten:
                # Extract text from handwritten region
                text = extract_text_from_region(image_pil, mask)
                if text:
                    handwritten_texts.append(text)
                    region_results.append({
                        "id": i,
                        "type": "handwritten",
                        "text": text,
                        "area": int(region['area'])
                    })
                    
                    # Mark handwritten regions in red
                    overlay[mask] = (overlay[mask] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
            else:
                # Mark printed regions in green
                overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
                region_results.append({
                    "id": i,
                    "type": "printed",
                    "text": "",
                    "area": int(region['area'])
                })
        
        # Combine all handwritten text
        combined_text = "\n".join(handwritten_texts)
        
        # Create overlay image
        overlay_image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_image.save(buffer, format='PNG')
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Original image
        buffer_orig = io.BytesIO()
        image_pil.save(buffer_orig, format='PNG')
        image_base64 = base64.b64encode(buffer_orig.getvalue()).decode()
        
        handwritten_count = sum(1 for r in region_results if r['type'] == 'handwritten')
        
        return JSONResponse({
            "success": True,
            "extracted_text": combined_text,
            "regions": region_results,
            "handwritten_regions": handwritten_count,
            "total_regions": len(regions),
            "overlay_image": f"data:image/png;base64,{overlay_base64}",
            "original_image": f"data:image/png;base64,{image_base64}",
            "method": "sam2" if sam2_mask_generator is not None else "simple"
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



