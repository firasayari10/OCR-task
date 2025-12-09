"""
Text Recognition Agent - Specialized in text recognition tasks.
"""

from typing import Dict, Any, List
import numpy as np
import logging
from .base_agent import BaseAgent, AgentResponse, AgentMessage, MessageRole

logger = logging.getLogger(__name__)


class TextRecognitionAgent(BaseAgent):
    """
    Agent specialized in text recognition from images.
    
    Capabilities:
    - OCR using Azure Vision API
    - Handwritten text recognition using TrOCR
    - Region-based text extraction
    - Text aggregation and formatting
    """
    
    def __init__(self):
        super().__init__(
            name="TextRecognitionAgent",
            description="Recognizes text from images using OCR and handwriting recognition models",
            system_prompt="""You are a text recognition specialist. Your job is to:
1. Extract printed text using Azure Vision OCR
2. Recognize handwritten text using TrOCR
3. Process multiple regions and aggregate results
4. Format and structure extracted text"""
        )
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a text recognition task.
        
        Args:
            task: Task description (e.g., "recognize text", "ocr regions")
            context: Must contain 'image' or 'regions' key
            
        Returns:
            AgentResponse with recognized text
        """
        try:
            self.add_message(AgentMessage(
                role=MessageRole.USER,
                content=task,
                metadata=context
            ))
            
            task_lower = task.lower()
            
            if "regions" in context:
                # Process multiple regions
                return await self._recognize_regions(context['regions'], context)
            elif "image" in context:
                # Process single image
                return await self._recognize_image(context['image'], context)
            else:
                return AgentResponse(
                    success=False,
                    error="No image or regions provided in context",
                    agent_name=self.name
                )
                
        except Exception as e:
            logger.error(f"TextRecognitionAgent error: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )
    
    async def _recognize_image(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Recognize text from a single image."""
        try:
            method = context.get('method', 'auto')  # auto, azure, trocr
            
            results = {}
            tools_used = []
            
            logger.info(f"Text recognition starting with method: {method}")
            logger.info(f"Available tools: {list(self.tools.keys())}")
            
            # Azure Vision OCR (good for printed text)
            if method in ['auto', 'azure'] and 'azure_vision_ocr' in self.tools:
                try:
                    logger.info("Calling Azure Vision OCR...")
                    azure_result = await self.use_tool("azure_vision_ocr", image=image)
                    results['azure'] = azure_result
                    tools_used.append('azure_vision_ocr')
                    logger.info(f"Azure OCR completed, result status: {azure_result.get('status', 'unknown')}")
                except Exception as e:
                    logger.error(f"Azure OCR failed: {e}", exc_info=True)
            else:
                logger.warning(f"Azure Vision OCR skipped - method: {method}, tool available: {'azure_vision_ocr' in self.tools}")
            
            # TrOCR (good for handwritten text)
            if method in ['auto', 'trocr'] and 'trocr_recognize' in self.tools:
                try:
                    logger.info("Calling TrOCR...")
                    trocr_result = await self.use_tool("trocr_recognize", image=image)
                    results['trocr'] = trocr_result
                    tools_used.append('trocr_recognize')
                    logger.info("TrOCR completed")
                except Exception as e:
                    logger.error(f"TrOCR failed: {e}", exc_info=True)
            
            # Extract text from results
            extracted_text = self._extract_text_from_results(results)
            logger.info(f"Extracted text length: {len(extracted_text)} characters")
            
            return AgentResponse(
                success=True,
                data={
                    "text": extracted_text,
                    "raw_results": results
                },
                metadata={
                    "methods_used": list(results.keys()),
                    "text_length": len(extracted_text)
                },
                agent_name=self.name,
                tools_used=tools_used
            )
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Text recognition failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _recognize_regions(self, regions: List[Dict[str, Any]], context: Dict[str, Any]) -> AgentResponse:
        """Recognize text from multiple image regions."""
        try:
            all_results = []
            total_text = []
            tools_used = []
            
            for i, region in enumerate(regions):
                region_image = region.get('image')
                if region_image is None:
                    continue
                
                # Recognize text in this region
                region_context = {**context, 'method': context.get('method', 'auto')}
                result = await self._recognize_image(region_image, region_context)
                
                if result.success:
                    text = result.data.get('text', '')
                    if text.strip():
                        all_results.append({
                            "region_id": region.get('id', i),
                            "bbox": region.get('bbox'),
                            "text": text,
                            "raw_results": result.data.get('raw_results', {})
                        })
                        total_text.append(text)
                        tools_used.extend(result.tools_used)
            
            # Combine all text
            combined_text = "\n".join(total_text)
            
            logger.info(f"Recognized text from {len(all_results)} regions")
            
            return AgentResponse(
                success=True,
                data={
                    "text": combined_text,
                    "regions": all_results
                },
                metadata={
                    "num_regions_processed": len(regions),
                    "num_regions_with_text": len(all_results),
                    "total_text_length": len(combined_text)
                },
                agent_name=self.name,
                tools_used=list(set(tools_used))
            )
            
        except Exception as e:
            logger.error(f"Region text recognition failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Region text recognition failed: {str(e)}",
                agent_name=self.name
            )
    
    def _extract_text_from_results(self, results: Dict[str, Any]) -> str:
        """Extract and combine text from various OCR results."""
        text_parts = []
        
        logger.info(f"Extracting text from results: {list(results.keys())}")
        
        # Azure Vision results
        if 'azure' in results:
            azure_data = results['azure']
            logger.info(f"Azure data keys: {list(azure_data.keys())}")
            logger.info(f"Azure status: {azure_data.get('status', 'no status')}")
            
            if azure_data.get('status') == 'succeeded':
                read_results = azure_data.get('analyzeResult', {}).get('readResults', [])
                logger.info(f"Found {len(read_results)} read results")
                
                for page_idx, page in enumerate(read_results):
                    lines = page.get('lines', [])
                    logger.info(f"Page {page_idx}: {len(lines)} lines")
                    for line in lines:
                        text = line.get('text', '')
                        if text:
                            text_parts.append(text)
                            logger.debug(f"  Line: {text}")
            else:
                logger.warning(f"Azure OCR not succeeded. Status: {azure_data.get('status')}")
        
        # TrOCR results
        if 'trocr' in results:
            trocr_text = results['trocr']
            if isinstance(trocr_text, str) and trocr_text.strip():
                text_parts.append(trocr_text)
                logger.info(f"TrOCR text: {trocr_text[:100]}...")
        
        final_text = "\n".join(text_parts)
        logger.info(f"Total extracted text: {len(final_text)} characters")
        return final_text
