from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os

app = FastAPI(title="Prescription OCR with SAM2 Segmentation")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for SAM2 model
sam2_model = None
sam2_predictor = None
sam2_mask_generator = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam2_model():
    """Load SAM2 model"""
    global sam2_model, sam2_predictor, sam2_mask_generator
    
    if sam2_model is not None:
        return
    
    try:
        # SAM2 configuration - using sam2_hiera_large model
        # You can change this to sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base, or sam2_hiera_large
        model_cfg = "sam2_hiera_large.yaml"
        checkpoint_path = "checkpoints/sam2_hiera_large.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                "Please download the model weights. See README.md for instructions."
            )
        
        print(f"Loading SAM2 model on {device}...")
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # Create automatic mask generator for better segmentation
        sam2_mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Filter out small regions
        )
        
        print("SAM2 model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_sam2_model()
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("⚠️  MODEL NOT FOUND")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nTo download the model, run:")
        print("  python download_model_requests.py")
        print("\nOr download manually from:")
        print("  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt")
        print("\nPlace it in: backend/checkpoints/sam2_hiera_large.pt")
        print(f"{'='*60}\n")
        print("⚠️  Server starting WITHOUT model - segmentation will not work!")
        print("   The /api/segment endpoint will return errors until model is loaded.\n")
    except Exception as e:
        print(f"Warning: Could not load SAM2 model on startup: {e}")
        print("The server will start but segmentation will not work until the model is loaded.")

@app.get("/")
async def root():
    return {"message": "Prescription OCR API with SAM2 Segmentation", "status": "running"}

@app.get("/health")
async def health_check():
    """Check if model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": sam2_model is not None,
        "mask_generator_loaded": sam2_mask_generator is not None,
        "device": device
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded image to numpy array"""
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to numpy array
    image_array = np.array(image)
    return image_array

def segment_text_regions(image: np.ndarray) -> Dict:
    """
    Use SAM2 automatic mask generation to segment all regions in the image.
    """
    global sam2_mask_generator
    
    if sam2_mask_generator is None:
        raise HTTPException(status_code=503, detail="SAM2 model not loaded")
    
    try:
        # Generate masks automatically
        masks = sam2_mask_generator.generate(image)
        
        if not masks:
            h, w = image.shape[:2]
            # Fallback: return the whole image as one segment
            return {
                "masks": [np.ones((h, w), dtype=bool)],
                "scores": [1.0],
                "regions": 1
            }
        
        # Sort masks by area (largest first) and take top regions
        masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Limit to top 15 regions to avoid too many small segments
        top_masks = masks_sorted[:15]
        
        mask_arrays = [m['segmentation'] for m in top_masks]
        scores = [m.get('predicted_iou', m.get('stability_score', 0.8)) for m in top_masks]
        
        return {
            "masks": mask_arrays,
            "scores": scores,
            "regions": len(top_masks)
        }
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

def classify_text_type(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Classify whether text in a region is handwritten or printed.
    This is a simplified heuristic - you may want to use a more sophisticated approach.
    """
    # Extract the masked region
    masked_region = image[mask]
    
    if len(masked_region) == 0:
        return "unknown"
    
    # Convert to grayscale for analysis
    if len(masked_region.shape) == 3:
        gray = cv2.cvtColor(masked_region.reshape(-1, masked_region.shape[-1]), cv2.COLOR_RGB2GRAY)
    else:
        gray = masked_region.flatten()
    
    # Simple heuristics (can be improved with ML models)
    # Handwritten text tends to have more variation in stroke width
    # and less uniform spacing
    
    # Calculate variance in pixel intensities
    variance = np.var(gray)
    
    # Calculate edge density (handwritten text has more irregular edges)
    edges = cv2.Canny(masked_region.reshape(image.shape[:2])[mask].reshape(-1, 1) if len(masked_region.shape) == 1 else cv2.cvtColor(masked_region.reshape(-1, masked_region.shape[-1]), cv2.COLOR_RGB2GRAY), 50, 150)
    edge_density = np.sum(edges > 0) / len(edges) if len(edges) > 0 else 0
    
    # Threshold-based classification (these are rough estimates)
    if variance > 2000 and edge_density > 0.1:
        return "handwritten"
    else:
        return "printed"

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    """
    Upload an image and get segmentation results with text type classification
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        # Perform segmentation
        segmentation_result = segment_text_regions(image)
        
        # Classify each region
        regions = []
        for i, mask in enumerate(segmentation_result["masks"]):
            text_type = classify_text_type(image, mask)
            
            # Create a colored mask for visualization
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = [255, 0, 0] if text_type == "handwritten" else [0, 255, 0]
            
            # Convert mask to base64 for frontend
            mask_image = Image.fromarray(colored_mask.astype(np.uint8))
            buffer = io.BytesIO()
            mask_image.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            regions.append({
                "id": i,
                "type": text_type,
                "score": float(segmentation_result["scores"][i]),
                "mask_image": f"data:image/png;base64,{mask_base64}"
            })
        
        # Create overlay image
        overlay = image.copy()
        for i, mask in enumerate(segmentation_result["masks"]):
            color = [255, 0, 0] if regions[i]["type"] == "handwritten" else [0, 255, 0]
            overlay[mask] = (overlay[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
        
        overlay_image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_image.save(buffer, format='PNG')
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "regions": regions,
            "overlay_image": f"data:image/png;base64,{overlay_base64}",
            "total_regions": len(regions),
            "handwritten_count": sum(1 for r in regions if r["type"] == "handwritten"),
            "printed_count": sum(1 for r in regions if r["type"] == "printed")
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

