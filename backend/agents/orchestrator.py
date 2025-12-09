"""
Orchestrator Agent - Top-level agent that can coordinate multiple agents and handle complex workflows.
"""

from typing import Dict, Any, List, Optional
import logging
from .base_agent import BaseAgent, AgentResponse, AgentMessage, MessageRole

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Top-level orchestrator agent that coordinates multiple specialized agents.
    
    This agent can:
    - Route requests to appropriate specialized agents
    - Coordinate multi-step workflows
    - Handle batch processing
    - Manage complex decision trees
    - Provide unified interface to the entire system
    """
    
    def __init__(self):
        super().__init__(
            name="OrchestratorAgent",
            description="Top-level orchestrator that routes tasks to specialized agents and coordinates workflows",
            system_prompt="""You are the system orchestrator. Your responsibilities:
1. Analyze incoming requests and determine which agent(s) to use
2. Coordinate complex multi-step workflows
3. Handle batch processing and parallel tasks
4. Provide fallback strategies when agents fail
5. Aggregate and format results from multiple agents
6. Maintain overall system state and context"""
        )
        self._routing_rules = {}
    
    def add_routing_rule(self, pattern: str, agent_name: str, priority: int = 0):
        """
        Add a routing rule for task delegation.
        
        Args:
            pattern: Keyword or pattern to match in task description
            agent_name: Name of agent to route to
            priority: Priority level (higher = higher priority)
        """
        if pattern not in self._routing_rules:
            self._routing_rules[pattern] = []
        self._routing_rules[pattern].append((agent_name, priority))
        self._routing_rules[pattern].sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Added routing rule: '{pattern}' -> {agent_name} (priority: {priority})")
    
    async def process(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a task by routing to appropriate agent(s).
        
        Args:
            task: Task description
            context: Task context and parameters
            
        Returns:
            AgentResponse with results
        """
        try:
            self.add_message(AgentMessage(
                role=MessageRole.USER,
                content=task,
                metadata=context
            ))
            
            # Determine task type
            task_type = context.get('task_type', self._infer_task_type(task))
            
            if task_type == 'batch':
                return await self._process_batch(task, context)
            elif task_type == 'workflow':
                return await self._process_workflow(task, context)
            else:
                return await self._process_single(task, context)
                
        except Exception as e:
            logger.error(f"OrchestratorAgent error: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                error=str(e),
                agent_name=self.name
            )
    
    def _infer_task_type(self, task: str) -> str:
        """Infer task type from task description."""
        task_lower = task.lower()
        
        if 'batch' in task_lower or 'multiple' in task_lower:
            return 'batch'
        elif 'workflow' in task_lower or 'pipeline' in task_lower:
            return 'workflow'
        else:
            return 'single'
    
    def _route_task(self, task: str) -> Optional[str]:
        """Route a task to the appropriate agent based on routing rules."""
        task_lower = task.lower()
        
        # Check routing rules
        for pattern, routes in self._routing_rules.items():
            if pattern.lower() in task_lower:
                # Return highest priority route
                return routes[0][0]
        
        # Default routing based on keywords
        if any(kw in task_lower for kw in ['ocr', 'prescription', 'extract', 'read', 'scan']):
            return 'OCRAgent'
        elif any(kw in task_lower for kw in ['segment', 'region', 'mask']):
            return 'SegmentationAgent'
        elif any(kw in task_lower for kw in ['text', 'recognize', 'handwriting']):
            return 'TextRecognitionAgent'
        elif any(kw in task_lower for kw in ['phi', 'filter', 'redact', 'hipaa']):
            return 'PHIFilterAgent'
        
        # Default to OCRAgent
        return 'OCRAgent'
    
    async def _process_single(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Process a single task by routing to appropriate agent."""
        try:
            # Determine target agent
            target_agent = context.get('agent') or self._route_task(task)
            
            if target_agent not in self.sub_agents:
                return AgentResponse(
                    success=False,
                    error=f"Agent '{target_agent}' not available",
                    agent_name=self.name
                )
            
            logger.info(f"Routing task to {target_agent}: {task[:100]}")
            
            # Delegate to target agent
            result = await self.delegate_to_agent(target_agent, task, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Single task processing failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Task processing failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _process_batch(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Process multiple tasks in batch."""
        try:
            tasks = context.get('tasks', [])
            if not tasks:
                return AgentResponse(
                    success=False,
                    error="No tasks provided for batch processing",
                    agent_name=self.name
                )
            
            results = []
            successful = 0
            failed = 0
            
            for i, task_item in enumerate(tasks):
                task_desc = task_item.get('task', '')
                task_context = task_item.get('context', {})
                
                logger.info(f"Processing batch task {i+1}/{len(tasks)}")
                
                result = await self._process_single(task_desc, task_context)
                results.append({
                    "task_id": task_item.get('id', i),
                    "task": task_desc,
                    "result": result.to_dict()
                })
                
                if result.success:
                    successful += 1
                else:
                    failed += 1
            
            return AgentResponse(
                success=True,
                data={
                    "results": results,
                    "summary": {
                        "total": len(tasks),
                        "successful": successful,
                        "failed": failed
                    }
                },
                metadata={
                    "batch_size": len(tasks),
                    "success_rate": successful / len(tasks) if tasks else 0
                },
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Batch processing failed: {str(e)}",
                agent_name=self.name
            )
    
    async def _process_workflow(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Process a multi-step workflow."""
        try:
            workflow_steps = context.get('workflow_steps', [])
            if not workflow_steps:
                # Default workflow for OCR
                workflow_steps = [
                    {"agent": "SegmentationAgent", "task": "segment image"},
                    {"agent": "TextRecognitionAgent", "task": "recognize text"},
                    {"agent": "PHIFilterAgent", "task": "filter phi"}
                ]
            
            workflow_context = context.copy()
            results = []
            
            for i, step in enumerate(workflow_steps):
                step_agent = step.get('agent')
                step_task = step.get('task')
                step_params = step.get('params', {})
                
                logger.info(f"Executing workflow step {i+1}/{len(workflow_steps)}: {step_agent}")
                
                # Merge parameters
                step_context = {**workflow_context, **step_params}
                
                # Execute step
                result = await self.delegate_to_agent(step_agent, step_task, step_context)
                
                results.append({
                    "step": i + 1,
                    "agent": step_agent,
                    "task": step_task,
                    "result": result.to_dict()
                })
                
                if not result.success:
                    logger.warning(f"Workflow step {i+1} failed: {result.error}")
                    if step.get('required', True):
                        return AgentResponse(
                            success=False,
                            error=f"Required workflow step {i+1} failed: {result.error}",
                            data={"completed_steps": results},
                            agent_name=self.name
                        )
                else:
                    # Pass data to next step
                    if result.data:
                        workflow_context.update(result.data)
            
            return AgentResponse(
                success=True,
                data={
                    "workflow_results": results,
                    "final_context": workflow_context
                },
                metadata={
                    "num_steps": len(workflow_steps),
                    "completed_steps": len(results)
                },
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"Workflow processing failed: {e}")
            return AgentResponse(
                success=False,
                error=f"Workflow processing failed: {str(e)}",
                agent_name=self.name
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        status = {
            "orchestrator": self.get_capabilities(),
            "agents": {}
        }
        
        for agent_name, agent in self.sub_agents.items():
            status["agents"][agent_name] = agent.get_capabilities()
        
        return status
