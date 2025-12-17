"""
Base agent class for the agentic OCR system.

Provides the foundation for all specialized agents with tool calling
and inter-agent communication capabilities.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in agent communication."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class AgentMessage:
    """Message in agent communication."""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class AgentResponse:
    """Response from an agent."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_name: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "agent_name": self.agent_name,
            "tools_used": self.tools_used
        }


class Tool:
    """Represents a tool that an agent can use."""
    
    def __init__(self, name: str, description: str, function: Callable, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        try:
            # Check if function is async
            import asyncio
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            raise


class BaseAgent:
    """
    Base class for all agents in the system.
    
    Agents can:
    - Use tools to accomplish tasks
    - Call other agents for specialized subtasks
    - Maintain conversation history
    - Return structured responses
    """
    
    def __init__(self, name: str, description: str, system_prompt: str = ""):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools: Dict[str, Tool] = {}
        self.sub_agents: Dict[str, 'BaseAgent'] = {}
        self.conversation_history: List[AgentMessage] = []
        
        # Add system message to history
        if system_prompt:
            self.conversation_history.append(
                AgentMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )
    
    def register_tool(self, tool: Tool):
        """Register a tool for this agent to use."""
        self.tools[tool.name] = tool
        logger.info(f"Agent {self.name}: Registered tool '{tool.name}'")
    
    def register_agent(self, agent: 'BaseAgent'):
        """Register a sub-agent that this agent can delegate to."""
        self.sub_agents[agent.name] = agent
        logger.info(f"Agent {self.name}: Registered sub-agent '{agent.name}'")
    
    def add_message(self, message: AgentMessage):
        """Add a message to conversation history."""
        self.conversation_history.append(message)
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a registered tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in agent {self.name}")
        
        logger.info(f"Agent {self.name}: Using tool '{tool_name}' with params: {list(kwargs.keys())}")
        tool = self.tools[tool_name]
        result = await tool.execute(**kwargs)
        
        # Log tool usage in conversation
        self.add_message(AgentMessage(
            role=MessageRole.TOOL,
            content=f"Used tool: {tool_name}",
            metadata={"tool": tool_name, "params": kwargs, "result": result}
        ))
        
        return result
    
    async def delegate_to_agent(self, agent_name: str, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Delegate a task to a sub-agent."""
        if agent_name not in self.sub_agents:
            raise ValueError(f"Agent '{agent_name}' not found in {self.name}'s sub-agents")
        
        logger.info(f"Agent {self.name}: Delegating to agent '{agent_name}': {task[:100]}")
        agent = self.sub_agents[agent_name]
        response = await agent.process(task, context or {})
        
        # Log delegation in conversation
        self.add_message(AgentMessage(
            role=MessageRole.ASSISTANT,
            content=f"Delegated to {agent_name}: {task}",
            metadata={"agent": agent_name, "task": task, "response": response.to_dict()}
        ))
        
        return response
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a task. This should be implemented by subclasses.
        
        Args:
            task: Description of the task to perform
            context: Additional context needed for the task
            
        Returns:
            AgentResponse with the result
        """
        raise NotImplementedError(f"Agent {self.name} must implement process()")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return a description of this agent's capabilities."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": list(self.tools.keys()),
            "sub_agents": list(self.sub_agents.keys())
        }
    
    def clear_history(self):
        """Clear conversation history (keeps system prompt)."""
        system_messages = [msg for msg in self.conversation_history if msg.role == MessageRole.SYSTEM]
        self.conversation_history = system_messages
