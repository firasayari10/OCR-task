"""
Segmentation Agent - Specialized in image segmentation tasks.
"""

from typing import Dict, Any
import numpy as np
import logging
from .base_agent import BaseAgent, AgentResponse, AgentMessage, MessageRole

logger = logging.getLogger(__name__)


class SegmentationAgent(BaseAgent):
    """
    Agent specialized in image segmentation.
    
    Capabilities:
    - Segment images into regions using SAM2
    - Extract regions from segmentation masks
    - Filter regions by size/quality
    """
    
    def __init__(self):
        super().__init__(
            name="SegmentationAgent",
            description="Segments images into distinct regions using SAM2 and extracts region information",
            system_prompt="""You are a segmentation specialist. Your job is to:
1. Segment images into distinct regions
2. Extract and validate regions
3. Provide region metadata (bounding boxes, areas, confidence scores)
4. Filter out low-quality or irrelevant regions"""
        )
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a segmentation task.
        
        Args:
            task: Task description (e.g., "segment image", "extract regions")
            context: Must contain 'image' key with numpy array
            
        Returns:
            AgentResponse with segmentation results
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
            
            # Determine task type
            task_lower = task.lower()
            
            if "segment" in task_lower or "mask" in task_lower:
                return await self._segment_image(image, context)
            elif "extract" in task_lower or "region" in task_lower:
                return await self._extract_regions(image, context)
            else:
                # Default: full segmentation pipeline
                return await self._full_segmentation(image, context)
                
        except Exception as e:
            logger.error(f"SegmentationAgent error: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )
    
    async def _segment_image(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Segment image into regions."""
        try:
            # Use SAM2 segmentation tool
            masks = await self.use_tool("sam2_segment", image=image)
            
            logger.info(f"Segmented image into {len(masks)} regions")
            
            return AgentResponse(
                success=True,
                data={"masks": masks},
                metadata={"num_masks": len(masks)},
                agent_name=self.name,
                tools_used=["sam2_segment"]
            )
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Segmentation failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _extract_regions(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Extract regions from image."""
        try:
            masks = context.get('masks')
            
            # If no masks provided, segment first
            if masks is None:
                seg_result = await self._segment_image(image, context)
                if not seg_result.success:
                    return seg_result
                masks = seg_result.data['masks']
            
            # Extract regions using tool
            regions = await self.use_tool("extract_regions", image=image, masks=masks)
            
            # Filter regions by quality
            min_area = context.get('min_area', 100)
            filtered_regions = [r for r in regions if r.get('area', 0) >= min_area]
            
            logger.info(f"Extracted {len(filtered_regions)} regions (filtered from {len(regions)})")
            
            return AgentResponse(
                success=True,
                data={"regions": filtered_regions},
                metadata={
                    "num_regions": len(filtered_regions),
                    "total_regions": len(regions),
                    "filtered_count": len(regions) - len(filtered_regions)
                },
                agent_name=self.name,
                tools_used=["extract_regions"]
            )
            
        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Region extraction failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _full_segmentation(self, image: np.ndarray, context: Dict[str, Any]) -> AgentResponse:
        """Complete segmentation pipeline: segment + extract regions."""
        try:
            # First segment
            seg_result = await self._segment_image(image, context)
            if not seg_result.success:
                return seg_result
            
            # Then extract regions
            context['masks'] = seg_result.data['masks']
            region_result = await self._extract_regions(image, context)
            
            if not region_result.success:
                return region_result
            
            # Combine results
            return AgentResponse(
                success=True,
                data={
                    "masks": seg_result.data['masks'],
                    "regions": region_result.data['regions']
                },
                metadata={
                    **seg_result.metadata,
                    **region_result.metadata
                },
                agent_name=self.name,
                tools_used=["sam2_segment", "extract_regions"]
            )
            
        except Exception as e:
            logger.error(f"Full segmentation failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Full segmentation failed: {str(e)}",
                agent_name=self.name
            )
