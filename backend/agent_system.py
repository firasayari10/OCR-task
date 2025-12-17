"""
Agent-based OCR system initialization and setup.

This module initializes all agents and tools for the OCR system.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
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


def download_sam2_if_needed() -> str:
    """
    Download SAM2 model from cloud storage if not present locally.
    
    Returns:
        Path to SAM2 checkpoint file
    """
    sam2_path = Path("checkpoints/sam2_hiera_large.pt")
    
    # If file exists locally, use it
    if sam2_path.exists():
        logger.info(f"Using local SAM2 checkpoint: {sam2_path}")
        return str(sam2_path)
    
    # Try to download from cloud
    blob_url = os.getenv("SAM2_BLOB_URL")
    hf_repo = os.getenv("SAM2_HF_REPO")
    
    if hf_repo:
        # Download from HuggingFace Hub
        try:
            logger.info(f"Downloading SAM2 from HuggingFace Hub: {hf_repo}")
            from huggingface_hub import hf_hub_download
            
            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename="sam2_hiera_large.pt",
                cache_dir="checkpoints"
            )
            logger.info(f"✓ SAM2 downloaded to {downloaded_path}")
            return downloaded_path
        except Exception as e:
            logger.warning(f"Failed to download from HF Hub: {e}")
    
    if blob_url:
        # Download from Azure Blob or S3
        try:
            logger.info(f"Downloading SAM2 from cloud storage...")
            sam2_path.parent.mkdir(parents=True, exist_ok=True)
            
            import requests
            response = requests.get(blob_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(sam2_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                            logger.info(f"Download progress: {percent:.1f}%")
            
            logger.info(f"✓ SAM2 downloaded to {sam2_path}")
            return str(sam2_path)
        except Exception as e:
            logger.error(f"Failed to download SAM2 from cloud: {e}")
    
    # Fallback: use default path (will fail if not present)
    logger.warning("SAM2 not found and no download URL configured. Segmentation may fail.")
    logger.warning("Set SAM2_BLOB_URL or SAM2_HF_REPO environment variable for production.")
    return str(sam2_path)


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
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        enable_sam2: bool = True,
        enable_trocr: bool = True,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None
    ):
        """
        Initialize the agent system with all components.
        
        Args:
            azure_endpoint: Azure Vision API endpoint
            azure_key: Azure Vision API key
            hf_token: HuggingFace API token
            enable_sam2: Whether to enable SAM2 segmentation
            enable_trocr: Whether to enable TrOCR
            sam2_checkpoint: Path to SAM2 checkpoint (auto-downloads if not provided)
            sam2_config: Path to SAM2 config
        """
        if self._initialized:
            logger.warning("Agent system already initialized")
            return
        
        logger.info("Initializing agent system...")
        
        # Download SAM2 if needed (production mode)
        if enable_sam2 and not sam2_checkpoint:
            sam2_checkpoint = download_sam2_if_needed()
        
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
                
                trocr_device: int = 0 if torch.cuda.is_available() else -1
                self.trocr_pipeline = pipeline(
                    "image-to-text",
                    model="microsoft/trocr-large-handwritten",
                    device=trocr_device
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
        azure_endpoint: Optional[str],
        azure_key: Optional[str],
        hf_token: Optional[str],
        enable_sam2: bool,
        enable_trocr: bool
    ):
        """Register all tools with appropriate agents."""
        
        # Segmentation tools
        if enable_sam2 and self.sam2_mask_generator:
            sam2_tool = create_sam2_segmentation_tool(self.sam2_mask_generator)
            if self.segmentation_agent:
                self.segmentation_agent.register_tool(sam2_tool)
        
        extract_tool = create_region_extraction_tool()
        if self.segmentation_agent:
            self.segmentation_agent.register_tool(extract_tool)
        
        # Text recognition tools
        if azure_endpoint and azure_key:
            azure_tool = create_azure_vision_ocr_tool(azure_endpoint, azure_key)
            if self.text_recognition_agent:
                self.text_recognition_agent.register_tool(azure_tool)
        
        if enable_trocr and self.trocr_pipeline:
            trocr_tool = create_trocr_tool(self.trocr_pipeline)
            if self.text_recognition_agent:
                self.text_recognition_agent.register_tool(trocr_tool)
        
        # PHI filtering tools
        phi_tool = create_phi_filter_tool(hf_token)
        if self.phi_filter_agent:
            self.phi_filter_agent.register_tool(phi_tool)
        
        # Image preprocessing (shared)
        preprocess_tool = create_image_preprocessing_tool()
        if self.segmentation_agent:
            self.segmentation_agent.register_tool(preprocess_tool)
        if self.text_recognition_agent:
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
        
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not available")
        
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
        if not self._initialized or not self.orchestrator:
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
        # TrOCR disabled - using Azure Vision API for OCR instead
        await _agent_system.initialize(
            azure_endpoint=os.getenv('AZURE_VISION_ENDPOINT'),
            azure_key=os.getenv('AZURE_VISION_KEY'),
            hf_token=os.getenv('HF_TOKEN'),
            enable_sam2=True,
            enable_trocr=False  # Disabled - using Azure Vision API
        )
    
    return _agent_system


async def shutdown_agent_system():
    """Shutdown and cleanup the agent system."""
    global _agent_system
    
    if _agent_system is not None:
        logger.info("Shutting down agent system...")
        # Cleanup if needed
        _agent_system = None
