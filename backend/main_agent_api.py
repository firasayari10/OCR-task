"""
Main API with Agent System Integration

This version uses the agent-based architecture for OCR processing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import agent system
from agent_system import get_agent_system

app = FastAPI(title="OCR Agent System API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent system instance
agent_system = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent system on startup."""
    global agent_system
    print("Initializing Agent System...")
    agent_system = await get_agent_system()
    status = agent_system.get_status()
    print(f"Agent System initialized: {status['initialized']}")
    print(f"SAM2 available: {status['models']['sam2_loaded']}")
    print(f"TrOCR available: {status['models']['trocr_loaded']}")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "OCR Agent System API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/api/agent-status",
            "process": "/api/process-image (POST)",
            "agent-process": "/api/agent/process (POST)",
            "batch": "/api/agent/batch (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global agent_system
    if agent_system is None:
        return {"status": "initializing"}
    
    status = agent_system.get_status()
    return {
        "status": "healthy",
        "agent_system_initialized": status['initialized'],
        "sam2_loaded": status['models']['sam2_loaded'],
        "trocr_loaded": status['models']['trocr_loaded']
    }


@app.get("/api/agent-status")
async def get_agent_status():
    """Get detailed agent system status."""
    global agent_system
    if agent_system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    status = agent_system.get_status()
    return status


@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...),
    mode: str = "full",
    filter_phi: bool = True,
    include_regions: bool = False
):
    """
    Process an image using the agent system.
    
    Args:
        file: Image file to process
        mode: Processing mode ('full', 'segment_only', 'ocr_only')
        filter_phi: Whether to filter PHI
        include_regions: Whether to include detailed region data
    
    Returns:
        JSON response with OCR results
    """
    global agent_system
    if agent_system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process through agent system
        result = await agent_system.process_image(
            image=image,
            mode=mode,
            filter_phi=filter_phi,
            include_regions=include_regions
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))
        
        # Format response for UI
        response_data = {
            "success": True,
            "mode": mode,
            "agent_used": result.get('agent_name'),
            "tools_used": result.get('tools_used', [])
        }
        
        data = result.get('data', {})
        
        # Handle different modes
        if mode == "segment_only":
            response_data["segmentation"] = data.get('segmentation', data)
            response_data["regions_detected"] = data.get('num_regions', 0)
            
        elif mode == "ocr_only":
            response_data["extracted_text"] = data.get('text', '')
            if filter_phi:
                response_data["redacted_text"] = data.get('redacted_text', '')
                response_data["phi_summary"] = data.get('phi_entities', [])
                response_data["phi_types"] = data.get('phi_summary', {})
                
        else:  # full mode
            # Segmentation info
            seg_data = data.get('segmentation', {})
            response_data["regions_detected"] = seg_data.get('num_regions', 0)
            
            # Text recognition
            text_data = data.get('text_recognition', {})
            response_data["extracted_text"] = text_data.get('text', '')
            
            # PHI filtering
            phi_data = data.get('phi_filtering', {})
            response_data["redacted_text"] = phi_data.get('redacted_text', '')
            response_data["phi_summary"] = phi_data.get('phi_entities', [])
            response_data["phi_types"] = phi_data.get('phi_summary', {})
            
            # Drug information
            drug_data = data.get('drug_information', {})
            response_data["medications"] = drug_data.get('medications', [])
            response_data["drug_alternatives"] = drug_data.get('drug_alternatives', [])
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode()
        response_data["original_image"] = f"data:image/png;base64,{image_base64}"
        
        # Create annotated image if regions available
        if mode in ["full", "segment_only"] and include_regions:
            annotated_image = create_annotated_image(image, data)
            if annotated_image is not None:
                _, buffer_ann = cv2.imencode('.png', annotated_image)
                ann_base64 = base64.b64encode(buffer_ann).decode()
                response_data["annotated_image"] = f"data:image/png;base64,{ann_base64}"
        
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/agent/process")
async def agent_process(
    file: UploadFile = File(...),
    task: str = "Process this prescription image",
    agent: Optional[str] = None,
    filter_phi: bool = True
):
    """
    Process image with custom task description and optional agent selection.
    
    This endpoint gives you direct control over the agent system.
    
    Args:
        file: Image file
        task: Task description (e.g., "Extract medical information")
        agent: Specific agent to use (optional, will auto-route if not specified)
        filter_phi: Whether to filter PHI
    """
    global agent_system
    if agent_system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process through orchestrator
        context = {
            "image": image,
            "filter_phi": filter_phi
        }
        
        if agent:
            context["agent"] = agent
        
        response = await agent_system.orchestrator.process(task, context)
        
        # Format response
        result = {
            "success": response.success,
            "agent_used": response.agent_name,
            "tools_used": response.tools_used,
            "data": response.data,
            "metadata": response.metadata
        }
        
        if not response.success:
            result["error"] = response.error
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Error in agent processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/batch")
async def batch_process(
    files: List[UploadFile] = File(...),
    task: str = "Process prescription image",
    filter_phi: bool = True
):
    """
    Batch process multiple images.
    
    Args:
        files: List of image files
        task: Task description for each image
        filter_phi: Whether to filter PHI
    """
    global agent_system
    if agent_system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        # Prepare batch tasks
        batch_tasks = []
        
        for i, file in enumerate(files):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                batch_tasks.append({
                    "id": i,
                    "task": task,
                    "context": {
                        "image": image,
                        "filter_phi": filter_phi,
                        "filename": file.filename
                    }
                })
        
        # Process batch
        response = await agent_system.orchestrator.process(
            task="Batch process images",
            context={
                "task_type": "batch",
                "tasks": batch_tasks
            }
        )
        
        return JSONResponse({
            "success": response.success,
            "results": response.data,
            "summary": response.metadata
        })
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phi/filter")
async def filter_phi_text(text: str):
    """
    Filter PHI from text directly (without image).
    
    Args:
        text: Text to filter
    """
    global agent_system
    if agent_system is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    
    try:
        phi_agent = agent_system.phi_filter_agent
        response = await phi_agent.process(
            task="Filter PHI from text",
            context={"text": text}
        )
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)
        
        return JSONResponse({
            "success": True,
            "redacted_text": response.data['redacted_text'],
            "phi_entities": response.data['phi_entities'],
            "phi_summary": response.data['phi_summary'],
            "num_entities": response.data.get('num_phi_entities', 0)
        })
        
    except Exception as e:
        print(f"Error filtering PHI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_annotated_image(image: np.ndarray, data: Dict) -> Optional[np.ndarray]:
    """Create an annotated image with bounding boxes."""
    try:
        annotated = image.copy()
        
        # Get regions from data
        regions = None
        if 'segmentation' in data:
            regions = data['segmentation'].get('regions')
        elif 'regions' in data:
            regions = data['regions']
        
        if not regions:
            return None
        
        # Draw bounding boxes
        for region in regions:
            bbox = region.get('bbox')
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add area text
                area = region.get('area', 0)
                cv2.putText(annotated, f"Area: {area:.0f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
