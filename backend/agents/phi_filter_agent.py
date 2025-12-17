"""
PHI Filter Agent - Specialized in detecting and redacting Protected Health Information.
"""

from typing import Dict, Any
import logging
from .base_agent import BaseAgent, AgentResponse, AgentMessage, MessageRole

logger = logging.getLogger(__name__)


class PHIFilterAgent(BaseAgent):
    """
    Agent specialized in PHI detection and filtering.
    
    Capabilities:
    - Detect various types of PHI (names, addresses, dates, etc.)
    - Redact sensitive information
    - Provide detailed PHI reports
    - Configure redaction strategies
    """
    
    def __init__(self):
        super().__init__(
            name="PHIFilterAgent",
            description="Detects and redacts Protected Health Information from text",
            system_prompt="""You are a PHI filtering specialist. Your job is to:
1. Detect all types of PHI in text (names, addresses, dates, IDs, etc.)
2. Redact sensitive information appropriately
3. Provide detailed reports of what was found
4. Ensure HIPAA compliance"""
        )
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a PHI filtering task.
        
        Args:
            task: Task description (e.g., "filter phi", "redact sensitive data")
            context: Must contain 'text' key with text to filter
            
        Returns:
            AgentResponse with filtered text and PHI report
        """
        try:
            self.add_message(AgentMessage(
                role=MessageRole.USER,
                content=task,
                metadata=context
            ))
            
            text = context.get('text')
            if text is None:
                return AgentResponse(
                    success=False,
                    error="No text provided in context",
                    agent_name=self.name
                )
            
            # Filter PHI using tool
            result = await self.use_tool("filter_phi", text=text)
            
            redacted_text = result.get('redacted_text', '')
            phi_list = result.get('phi', [])
            
            # Generate PHI summary
            phi_summary = self._generate_phi_summary(phi_list)
            
            logger.info(f"Filtered {len(phi_list)} PHI entities from text")
            
            return AgentResponse(
                success=True,
                data={
                    "redacted_text": redacted_text,
                    "phi_entities": phi_list,
                    "phi_summary": phi_summary,
                    "original_length": len(text),
                    "redacted_length": len(redacted_text)
                },
                metadata={
                    "num_phi_entities": len(phi_list),
                    "phi_types": list(phi_summary.keys())
                },
                agent_name=self.name,
                tools_used=["filter_phi"]
            )
            
        except Exception as e:
            logger.error(f"PHIFilterAgent error: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )
    
    def _generate_phi_summary(self, phi_list: list) -> Dict[str, int]:
        """Generate a summary of PHI types found."""
        summary: Dict[str, Any] = {}
        for phi_item in phi_list:
            phi_type = phi_item.get('type', 'UNKNOWN')
            summary[phi_type] = summary.get(phi_type, 0) + 1
        return summary
