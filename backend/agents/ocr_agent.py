"""
OCR Agent - Main agent for complete OCR pipeline.
"""

from typing import Dict, Any
import numpy as np
import logging
from .base_agent import BaseAgent, AgentResponse, AgentMessage, MessageRole

logger = logging.getLogger(__name__)


class OCRAgent(BaseAgent):
    """
    Main OCR Agent that coordinates the complete OCR pipeline.
    
    This agent can delegate tasks to specialized sub-agents:
    - SegmentationAgent: For image segmentation
    - TextRecognitionAgent: For text extraction
    - PHIFilterAgent: For PHI filtering
    
    It provides a complete end-to-end OCR solution.
    """
    
    def __init__(self):
        super().__init__(
            name="OCRAgent",
            description="Main OCR agent that coordinates segmentation, text recognition, and PHI filtering",
            system_prompt="""You are the main OCR coordinator. Your job is to:
1. Analyze incoming OCR requests
2. Delegate segmentation tasks to SegmentationAgent
3. Delegate text recognition to TextRecognitionAgent
4. Delegate PHI filtering to PHIFilterAgent
5. Combine and format final results
6. Handle errors and provide fallbacks"""
        )
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a complete OCR task.
        
        Args:
            task: Task description (e.g., "process prescription", "extract text")
            context: Must contain 'image' key with numpy array
            
        Returns:
            AgentResponse with complete OCR results
        """
        try:
            self.add_message(AgentMessage(
                role=MessageRole.USER,
                content=task,
                metadata=context
            ))
            
            image = context.get('image')
            if image is None:
                return AgentResponse(
                    success=False,
                    error="No image provided in context",
                    agent_name=self.name
                )
            
            # Determine processing mode
            mode = context.get('mode', 'full')  # full, segment_only, ocr_only
            
            if mode == 'segment_only':
                return await self._process_segmentation_only(image, context)
            elif mode == 'ocr_only':
                return await self._process_ocr_only(image, context)
            else:
                return await self._process_full_pipeline(image, context)
                
        except Exception as e:
            logger.error(f"OCRAgent error: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )
    
    async def _process_segmentation_only(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Process only segmentation."""
        try:
            seg_result = await self.delegate_to_agent(
                "SegmentationAgent",
                "Segment this image into regions",
                {"image": image, **context}
            )
            
            return seg_result
            
        except Exception as e:
            logger.error(f"Segmentation only failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Segmentation failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _process_ocr_only(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Process only OCR without segmentation."""
        try:
            # Recognize text
            ocr_result = await self.delegate_to_agent(
                "TextRecognitionAgent",
                "Recognize all text in this image",
                {"image": image, **context}
            )
            
            if not ocr_result.success:
                return ocr_result
            
            # Optionally filter PHI
            if context.get('filter_phi', True):
                text = ocr_result.data.get('text', '')
                phi_result = await self.delegate_to_agent(
                    "PHIFilterAgent",
                    "Filter PHI from this text",
                    {"text": text}
                )
                
                if phi_result.success:
                    return AgentResponse(
                        success=True,
                        data={
                            **ocr_result.data,
                            **phi_result.data
                        },
                        metadata={
                            **ocr_result.metadata,
                            **phi_result.metadata
                        },
                        agent_name=self.name,
                        tools_used=ocr_result.tools_used + phi_result.tools_used
                    )
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"OCR only failed: {e}")
            return AgentResponse(
                success=False,
                error=f"OCR failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _process_full_pipeline(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Process complete OCR pipeline: segment -> recognize -> filter."""
        try:
            tools_used = []
            
            # Try to segment image (optional step)
            logger.info("Step 1: Attempting segmentation...")
            seg_result = await self.delegate_to_agent(
                "SegmentationAgent",
                "Segment this image and extract regions",
                {"image": image, **context}
            )
            
            regions_detected = 0
            if seg_result.success:
                tools_used.extend(seg_result.tools_used)
                regions = seg_result.data.get('regions', [])
                regions_detected = len(regions)
                logger.info(f"Segmentation found {regions_detected} regions")
            else:
                logger.warning("Segmentation failed, will use full image OCR")
            
            # Always do OCR on the full image (not just regions)
            logger.info("Step 2: Recognizing text from full image...")
            
            ocr_result = await self.delegate_to_agent(
                "TextRecognitionAgent",
                "Recognize all text in this image",
                {"image": image, **context}
            )
            
            if not ocr_result.success:
                logger.error(f"Text recognition failed: {ocr_result.error}")
                return AgentResponse(
                    success=False,
                    error=f"Text recognition failed: {ocr_result.error}",
                    agent_name=self.name
                )
            
            tools_used.extend(ocr_result.tools_used)
            extracted_text = ocr_result.data.get('text', '')
            logger.info(f"Extracted {len(extracted_text)} characters of text")
            
            # Step 3: Filter PHI (if requested)
            phi_result = None
            redacted_text = extracted_text
            if context.get('filter_phi', True) and extracted_text.strip():
                logger.info("Step 3: Filtering PHI...")
                phi_result = await self.delegate_to_agent(
                    "PHIFilterAgent",
                    "Filter PHI from extracted text",
                    {"text": extracted_text}
                )
                
                if phi_result.success:
                    tools_used.extend(phi_result.tools_used)
                    redacted_text = phi_result.data.get('redacted_text', extracted_text)
            
            # Step 4: Extract medications and get drug information (if requested)
            drug_info_result = None
            if context.get('extract_drugs', True) and redacted_text.strip():
                logger.info("Step 4: Extracting medications and querying drug databases...")
                drug_info_result = await self.delegate_to_agent(
                    "DrugInformationAgent",
                    "Extract medications and find drug alternatives",
                    {"text": redacted_text}  # Use PHI-filtered text for privacy
                )
                
                if drug_info_result.success:
                    tools_used.extend(drug_info_result.tools_used)
            
            # Combine all results
            final_data = {
                "segmentation": {
                    "num_regions": regions_detected,
                    "regions": None
                },
                "text_recognition": {
                    "text": extracted_text,
                    "raw_results": ocr_result.data.get('raw_results', {})
                }
            }
            
            if phi_result and phi_result.success:
                final_data["phi_filtering"] = phi_result.data
            else:
                final_data["phi_filtering"] = {
                    "redacted_text": extracted_text,
                    "phi_entities": [],
                    "phi_summary": {}
                }
            
            if drug_info_result and drug_info_result.success:
                final_data["drug_information"] = drug_info_result.data
            
            logger.info("OCR pipeline completed successfully")
            
            return AgentResponse(
                success=True,
                data=final_data,
                metadata={
                    "pipeline_steps": ["segmentation", "text_recognition", "phi_filtering", "drug_information"],
                    "num_regions": regions_detected,
                    "text_length": len(extracted_text),
                    "phi_filtered": phi_result is not None and phi_result.success,
                    "medications_found": drug_info_result.data.get('total_medications', 0) if drug_info_result and drug_info_result.success else 0
                },
                agent_name=self.name,
                tools_used=tools_used
            )
            
        except Exception as e:
            logger.error(f"Full pipeline failed: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=f"Pipeline failed: {str(e)}",
                agent_name=self.name
            )
