"""
Agent-based OCR system initialization and setup.

This module initializes all agents and tools for the OCR system.
"""

import logging
import os
from typing import Dict, Any, Optional
import torch
import numpy as np

# Import agents
from agents import (
    OrchestratorAgent,
    OCRAgent,
    SegmentationAgent,
    TextRecognitionAgent,
    PHIFilterAgent
)
from agents.drug_information_agent import DrugInformationAgent

# Import tools
from agents.tools import (
    create_sam2_segmentation_tool,
    create_azure_vision_ocr_tool,
    create_trocr_tool,
    create_phi_filter_tool,
    create_image_preprocessing_tool,
    create_region_extraction_tool
)

logger = logging.getLogger(__name__)


class AgentSystem:
    """
    Main agent system that initializes and manages all agents and tools.
    """
    
    def __init__(self):
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.ocr_agent: Optional[OCRAgent] = None
        self.segmentation_agent: Optional[SegmentationAgent] = None
        self.text_recognition_agent: Optional[TextRecognitionAgent] = None
        self.phi_filter_agent: Optional[PHIFilterAgent] = None
        self.drug_information_agent: Optional[DrugInformationAgent] = None
        
        self.sam2_model = None
        self.sam2_mask_generator = None
        self.trocr_pipeline = None
        
        self._initialized = False
    
    async def initialize(
        self,
        azure_endpoint: str = None,
        azure_key: str = None,
        hf_token: str = None,
        enable_sam2: bool = True,
        enable_trocr: bool = True,
        sam2_checkpoint: str = None,
        sam2_config: str = None
    ):
        """
        Initialize the agent system with all components.
        
        Args:
            azure_endpoint: Azure Vision API endpoint
            azure_key: Azure Vision API key
            hf_token: HuggingFace API token
            enable_sam2: Whether to enable SAM2 segmentation
            enable_trocr: Whether to enable TrOCR
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_config: Path to SAM2 config
        """
        if self._initialized:
            logger.warning("Agent system already initialized")
            return
        
        logger.info("Initializing agent system...")
        
        # Create agents
        self.orchestrator = OrchestratorAgent()
        self.ocr_agent = OCRAgent()
        self.segmentation_agent = SegmentationAgent()
        self.text_recognition_agent = TextRecognitionAgent()
        self.phi_filter_agent = PHIFilterAgent()
        self.drug_information_agent = DrugInformationAgent()
        
        # Initialize SAM2 if enabled
        if enable_sam2:
            try:
                logger.info("Loading SAM2 model...")
                from sam2.build_sam import build_sam2
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Default paths if not provided
                if sam2_checkpoint is None:
                    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
                if sam2_config is None:
                    # Use the full config path as expected by Hydra
                    sam2_config = "configs/sam2/sam2_hiera_l.yaml"
                
                # Verify checkpoint exists
                if not os.path.exists(sam2_checkpoint):
                    raise FileNotFoundError(f"SAM2 checkpoint not found at: {sam2_checkpoint}")
                
                logger.info(f"Loading SAM2 from checkpoint: {sam2_checkpoint}")
                logger.info(f"Using config: {sam2_config}")
                
                # Build SAM2 model
                self.sam2_model = build_sam2(
                    config_file=sam2_config,
                    ckpt_path=sam2_checkpoint,
                    device=device,
                    apply_postprocessing=False
                )
                
                self.sam2_mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)
                
                logger.info("✓ SAM2 model loaded successfully!")
                enable_sam2 = True
            except Exception as e:
                logger.error(f"Failed to load SAM2: {e}", exc_info=True)
                logger.warning("Segmentation will use fallback method.")
                enable_sam2 = False
                self.sam2_model = None
                self.sam2_mask_generator = None
        
        # Initialize TrOCR if enabled
        if enable_trocr:
            try:
                logger.info("Loading TrOCR model...")
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline
                
                device = 0 if torch.cuda.is_available() else -1
                self.trocr_pipeline = pipeline(
                    "image-to-text",
                    model="microsoft/trocr-large-handwritten",
                    device=device
                )
                
                logger.info("TrOCR model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load TrOCR: {e}. Handwriting recognition will be limited.")
                enable_trocr = False
        
        # Create and register tools
        self._register_tools(
            azure_endpoint=azure_endpoint,
            azure_key=azure_key,
            hf_token=hf_token,
            enable_sam2=enable_sam2,
            enable_trocr=enable_trocr
        )
        
        # Build agent hierarchy
        self._build_agent_hierarchy()
        
        # Setup routing rules
        self._setup_routing()
        
        self._initialized = True
        logger.info("Agent system initialized successfully")
    
    def _register_tools(
        self,
        azure_endpoint: str,
        azure_key: str,
        hf_token: str,
        enable_sam2: bool,
        enable_trocr: bool
    ):
        """Register all tools with appropriate agents."""
        
        # Segmentation tools
        if enable_sam2:
            sam2_tool = create_sam2_segmentation_tool(self.sam2_mask_generator)
            self.segmentation_agent.register_tool(sam2_tool)
        
        extract_tool = create_region_extraction_tool()
        self.segmentation_agent.register_tool(extract_tool)
        
        # Text recognition tools
        if azure_endpoint and azure_key:
            azure_tool = create_azure_vision_ocr_tool(azure_endpoint, azure_key)
            self.text_recognition_agent.register_tool(azure_tool)
        
        if enable_trocr:
            trocr_tool = create_trocr_tool(self.trocr_pipeline)
            self.text_recognition_agent.register_tool(trocr_tool)
        
        # PHI filtering tools
        phi_tool = create_phi_filter_tool(hf_token)
        self.phi_filter_agent.register_tool(phi_tool)
        
        # Image preprocessing (shared)
        preprocess_tool = create_image_preprocessing_tool()
        self.segmentation_agent.register_tool(preprocess_tool)
        self.text_recognition_agent.register_tool(preprocess_tool)
        
        logger.info("Tools registered successfully")
    
    def _build_agent_hierarchy(self):
        """Build the agent hierarchy and relationships."""
        
        # OCRAgent manages specialized agents
        self.ocr_agent.register_agent(self.segmentation_agent)
        self.ocr_agent.register_agent(self.text_recognition_agent)
        self.ocr_agent.register_agent(self.phi_filter_agent)
        self.ocr_agent.register_agent(self.drug_information_agent)
        
        # Orchestrator manages all agents
        self.orchestrator.register_agent(self.ocr_agent)
        self.orchestrator.register_agent(self.segmentation_agent)
        self.orchestrator.register_agent(self.text_recognition_agent)
        self.orchestrator.register_agent(self.phi_filter_agent)
        self.orchestrator.register_agent(self.drug_information_agent)
        
        logger.info("Agent hierarchy established")
    
    def _setup_routing(self):
        """Setup routing rules for the orchestrator."""
        
        self.orchestrator.add_routing_rule("prescription", "OCRAgent", priority=10)
        self.orchestrator.add_routing_rule("medical", "OCRAgent", priority=10)
        self.orchestrator.add_routing_rule("document", "OCRAgent", priority=5)
        self.orchestrator.add_routing_rule("segment", "SegmentationAgent", priority=10)
        self.orchestrator.add_routing_rule("recognize", "TextRecognitionAgent", priority=8)
        self.orchestrator.add_routing_rule("phi", "PHIFilterAgent", priority=10)
        self.orchestrator.add_routing_rule("hipaa", "PHIFilterAgent", priority=10)
        self.orchestrator.add_routing_rule("drug", "DrugInformationAgent", priority=10)
        self.orchestrator.add_routing_rule("medication", "DrugInformationAgent", priority=10)
        self.orchestrator.add_routing_rule("alternative", "DrugInformationAgent", priority=8)
        
        logger.info("Routing rules configured")
    
    async def process_image(
        self,
        image: np.ndarray,
        mode: str = "full",
        filter_phi: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an image through the agent system.
        
        Args:
            image: Input image as numpy array
            mode: Processing mode ('full', 'segment_only', 'ocr_only')
            filter_phi: Whether to filter PHI
            **kwargs: Additional parameters
            
        Returns:
            Processing results dictionary
        """
        if not self._initialized:
            raise RuntimeError("Agent system not initialized. Call initialize() first.")
        
        context = {
            "image": image,
            "mode": mode,
            "filter_phi": filter_phi,
            **kwargs
        }
        
        # Route through orchestrator
        response = await self.orchestrator.process(
            task=f"Process image with mode: {mode}",
            context=context
        )
        
        return response.to_dict()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status and capabilities."""
        if not self._initialized:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "system": self.orchestrator.get_system_status(),
            "models": {
                "sam2_loaded": self.sam2_model is not None,
                "trocr_loaded": self.trocr_pipeline is not None
            }
        }


# Global agent system instance
_agent_system: Optional[AgentSystem] = None


async def get_agent_system() -> AgentSystem:
    """Get or create the global agent system instance."""
    global _agent_system
    
    if _agent_system is None:
        _agent_system = AgentSystem()
        
        # Initialize with environment variables
        await _agent_system.initialize(
            azure_endpoint=os.getenv('AZURE_VISION_ENDPOINT'),
            azure_key=os.getenv('AZURE_VISION_KEY'),
            hf_token=os.getenv('HF_TOKEN'),
            enable_sam2=True,
            enable_trocr=True
        )
    
    return _agent_system


async def shutdown_agent_system():
    """Shutdown and cleanup the agent system."""
    global _agent_system
    
    if _agent_system is not None:
        logger.info("Shutting down agent system...")
        # Cleanup if needed
        _agent_system = None
