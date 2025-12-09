"""
Agent system for OCR processing.

This package provides an agentic architecture where specialized agents
can call tools and other agents to accomplish complex tasks.
"""

from .base_agent import BaseAgent, AgentResponse, AgentMessage
from .orchestrator import OrchestratorAgent
from .ocr_agent import OCRAgent
from .segmentation_agent import SegmentationAgent
from .text_recognition_agent import TextRecognitionAgent
from .phi_filter_agent import PHIFilterAgent

__all__ = [
    'BaseAgent',
    'AgentResponse', 
    'AgentMessage',
    'OrchestratorAgent',
    'OCRAgent',
    'SegmentationAgent',
    'TextRecognitionAgent',
    'PHIFilterAgent'
]
