"""
Tool definitions for the agent system.

This module contains all the tools that agents can use to interact with
OCR models, image processing, and external APIs.
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import base64
import io
import torch
import requests
import os
from .base_agent import Tool


def create_sam2_segmentation_tool(sam2_mask_generator) -> Tool:
    """Create a tool for SAM2 image segmentation."""
    
    def sam2_segment(image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """
        Segment an image using SAM2.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of segment masks with metadata
        """
        if sam2_mask_generator is None:
            raise ValueError("SAM2 mask generator not initialized")
        
        # Generate masks
        masks = sam2_mask_generator.generate(image)
        
        return masks
    
    return Tool(
        name="sam2_segment",
        description="Segment an image into regions using SAM2 model",
        function=sam2_segment,
        parameters={
            "image": {"type": "np.ndarray", "description": "Input image"},
        }
    )


def create_azure_vision_ocr_tool(endpoint: str, api_key: str) -> Tool:
    """Create a tool for Azure Vision OCR."""
    
    def azure_vision_ocr(image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform OCR using Azure Vision API.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            OCR results with text and bounding boxes
        """
        # Convert image to bytes
        _, buffer = cv2.imencode('.png', image)
        image_bytes = buffer.tobytes()
        
        # Call Azure Vision API
        headers = {
            'Ocp-Apim-Subscription-Key': api_key,
            'Content-Type': 'application/octet-stream'
        }
        
        url = f"{endpoint}/vision/v3.2/read/analyze"
        
        response = requests.post(url, headers=headers, data=image_bytes)
        response.raise_for_status()
        
        # Get operation location
        operation_location = response.headers.get('Operation-Location')
        
        # Poll for results
        import time
        max_attempts = 10
        for _ in range(max_attempts):
            time.sleep(1)
            result_response = requests.get(operation_location, headers={'Ocp-Apim-Subscription-Key': api_key})
            result = result_response.json()
            
            if result.get('status') in ['succeeded', 'failed']:
                break
        
        return result
    
    return Tool(
        name="azure_vision_ocr",
        description="Extract text from images using Azure Vision OCR",
        function=azure_vision_ocr,
        parameters={
            "image": {"type": "np.ndarray", "description": "Input image"},
        }
    )


def create_trocr_tool(model_pipeline) -> Tool:
    """Create a tool for TrOCR text recognition."""
    
    def trocr_recognize(image: np.ndarray, **kwargs) -> str:
        """
        Recognize text using TrOCR model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Recognized text
        """
        if model_pipeline is None:
            raise ValueError("TrOCR model not initialized")
        
        # Convert numpy array to PIL Image
        pil_image: Image.Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image  # type: ignore
        
        # Run TrOCR
        result = model_pipeline(pil_image)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        
        return ""
    
    return Tool(
        name="trocr_recognize",
        description="Recognize handwritten text using TrOCR model",
        function=trocr_recognize,
        parameters={
            "image": {"type": "np.ndarray", "description": "Input image"},
        }
    )


def create_phi_filter_tool(hf_token: Optional[str] = None) -> Tool:
    """Create a tool for PHI filtering."""
    
    def filter_phi(text: str, **kwargs) -> Dict[str, Any]:
        """
        Filter Protected Health Information from text.
        
        Args:
            text: Input text to filter
            
        Returns:
            Dictionary with redacted text and list of PHI entities found
        """
        from huggingface_hub import InferenceClient
        import re
        
        phi_spans = []
        
        if not text:
            return {"redacted_text": "", "phi": []}
        
        # NER with HuggingFace
        if hf_token:
            try:
                client = InferenceClient(token=hf_token)
                
                # Try ClinicalBERT embeddings
                try:
                    _ = client.feature_extraction(text, model="emilyalsentzer/Bio_ClinicalBERT")
                except Exception as e:
                    print(f"ClinicalBERT embeddings failed: {e}")
                
                # NER
                try:
                    ner_model = os.getenv('HF_NER_MODEL', 'dslim/bert-base-NER')
                    ner_results = client.token_classification(text, model=ner_model)
                    
                    grouped_entities = []
                    current_entity = None
                    
                    for ent in ner_results:
                        entity_type = ent.get('entity_group') or ent.get('entity', '')
                        entity_type_upper = str(entity_type).upper().replace('B-', '').replace('I-', '')
                        
                        if entity_type_upper in ("PER", "PERSON", "ORG", "LOC", "MISC"):
                            if current_entity and current_entity['type'] == entity_type_upper and ent.get('start') <= current_entity['end'] + 2:
                                current_entity['end'] = ent.get('end')
                                current_entity['word'] += ' ' + ent.get('word', '')
                            else:
                                if current_entity:
                                    grouped_entities.append(current_entity)
                                current_entity = {
                                    'type': entity_type_upper,
                                    'start': ent.get('start'),
                                    'end': ent.get('end'),
                                    'word': ent.get('word', '')
                                }
                    
                    if current_entity:
                        grouped_entities.append(current_entity)
                    
                    for ent_item in grouped_entities:
                        ent_dict = dict(ent_item) if isinstance(ent_item, dict) else ent_item  # Convert to dict to avoid type issues
                        phi_spans.append((ent_dict['start'], ent_dict['end'], ent_dict['type'], ent_dict['word']))
                        
                except Exception as e:
                    print(f"NER failed: {e}")
            except Exception as e:
                print(f"HF client error: {e}")
        
        # Regex-based PHI detection
        # Names
        for m in re.finditer(r"(?:Name|Patient\s*Name|Patient):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+?)(?:\s+Address|$|\s*\n)", text, flags=re.IGNORECASE):
            phi_spans.append((m.start(1), m.end(1), 'NAME', m.group(1)))
        
        # Addresses
        for m in re.finditer(r"(?:Address|Street|Location):\s*([A-Z0-9][^\n]{5,60}?)(?:\s+Age|Sex|Date|$|\s*\n)", text, flags=re.IGNORECASE):
            phi_spans.append((m.start(1), m.end(1), 'ADDRESS', m.group(1)))
        
        # Ages
        for m in re.finditer(r"(?:Age):\s*(\d{1,3})", text, flags=re.IGNORECASE):
            phi_spans.append((m.start(1), m.end(1), 'AGE', m.group(1)))
        
        # Dates
        for m in re.finditer(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", text):
            phi_spans.append((m.start(), m.end(), 'DATE', m.group(0)))
        
        for m in re.finditer(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b", text, flags=re.IGNORECASE):
            phi_spans.append((m.start(), m.end(), 'DATE', m.group(0)))
        
        # License numbers
        for m in re.finditer(r"(?:Lic\.?\s*No\.?|License\s*No\.?|PTR\s*No\.?|S2\s*No\.?)[\s:]*(\d+)", text, flags=re.IGNORECASE):
            phi_spans.append((m.start(1), m.end(1), 'LICENSE', m.group(1)))
        
        # Emails
        for m in re.finditer(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", text):
            phi_spans.append((m.start(), m.end(), 'EMAIL', m.group(0)))
        
        # Phone numbers
        for m in re.finditer(r"\b(?:\+\d{1,3}[- ]?)?(?:\(\d{2,4}\)|\d{2,4})[- ]?\d{3,4}[- ]?\d{3,4}\b", text):
            phi_spans.append((m.start(), m.end(), 'PHONE', m.group(0)))
        
        # IDs
        for m in re.finditer(r"\b\d{6,}\b", text):
            context = text[max(0, m.start()-20):m.end()+20]
            if re.search(r"\b(MRN|ID|Patient|Acct|Account|Record|No\.?)\b", context, flags=re.IGNORECASE):
                phi_spans.append((m.start(), m.end(), 'ID', m.group(0)))
        
        # SSN
        for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text):
            phi_spans.append((m.start(), m.end(), 'SSN', m.group(0)))
        
        # Sort and deduplicate
        phi_spans = sorted(set(phi_spans), key=lambda x: x[0])
        
        # Redact text
        redacted = list(text)
        phi_list = []
        
        for start, end, label, original_text in phi_spans:
            redacted_span = f"[{label}_REDACTED]"
            for i in range(start, min(end, len(redacted))):
                redacted[i] = ''
            if start < len(redacted):
                redacted[start] = redacted_span
            
            phi_list.append({
                "type": label,
                "original": original_text,
                "start": start,
                "end": end
            })
        
        redacted_text = ''.join(redacted)
        
        return {
            "redacted_text": redacted_text,
            "phi": phi_list
        }
    
    return Tool(
        name="filter_phi",
        description="Detect and redact Protected Health Information from text",
        function=filter_phi,
        parameters={
            "text": {"type": "str", "description": "Input text to filter"},
        }
    )


def create_image_preprocessing_tool() -> Tool:
    """Create a tool for image preprocessing."""
    
    def preprocess_image(image: np.ndarray, operations: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        """
        Preprocess an image with various operations.
        
        Args:
            image: Input image
            operations: List of operations to apply (e.g., ['grayscale', 'denoise', 'threshold'])
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        if operations is None:
            operations = []
        
        for op in operations:
            if op == 'grayscale':
                if len(result.shape) == 3:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            elif op == 'denoise':
                if len(result.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(result)
                else:
                    result = cv2.fastNlMeansDenoising(result)
            elif op == 'threshold':
                if len(result.shape) == 3:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif op == 'enhance_contrast':
                if len(result.shape) == 3:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                result = cv2.equalizeHist(result)
        
        return result
    
    return Tool(
        name="preprocess_image",
        description="Apply preprocessing operations to images",
        function=preprocess_image,
        parameters={
            "image": {"type": "np.ndarray", "description": "Input image"},
            "operations": {"type": "List[str]", "description": "List of preprocessing operations"},
        }
    )


def create_region_extraction_tool() -> Tool:
    """Create a tool for extracting regions from images."""
    
    def extract_regions(image: np.ndarray, masks: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract image regions based on masks.
        
        Args:
            image: Input image
            masks: List of mask dictionaries
            
        Returns:
            List of extracted regions with metadata
        """
        if masks is None:
            # Fallback: simple contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                region_img = image[y:y+h, x:x+w]
                
                regions.append({
                    "id": i,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "image": region_img
                })
            
            return regions
        
        # Extract regions from SAM2 masks
        regions = []
        for i, mask_data in enumerate(masks):
            mask = mask_data.get('segmentation')
            if mask is None:
                continue
            
            # Get bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
            
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # Extract region
            region_img = image[y_min:y_max+1, x_min:x_max+1]
            
            regions.append({
                "id": i,
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                "area": float(mask_data.get('area', 0)),
                "image": region_img,
                "confidence": float(mask_data.get('predicted_iou', 0))
            })
        
        return regions
    
    return Tool(
        name="extract_regions",
        description="Extract image regions from segmentation masks",
        function=extract_regions,
        parameters={
            "image": {"type": "np.ndarray", "description": "Input image"},
            "masks": {"type": "List[Dict]", "description": "Segmentation masks"},
        }
    )
