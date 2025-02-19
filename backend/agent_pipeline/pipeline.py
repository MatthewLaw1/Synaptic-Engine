"""
Main pipeline for converting thoughts to video prompts.
Uses a three-layer agent system for detailed, emotionally-aware video generation.
"""

from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from .base import BaseLLMPipeline
from .agents import ExpansionAgents, EvaluatorAgent, VideoPromptGenerator

# Load environment variables
load_dotenv()

class ThoughtPipeline(BaseLLMPipeline):
    """Handles the conversion of thoughts to video-ready prompts."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Initialize the pipeline.
        
        Args:
            model_name: Optional model name override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        """
        super().__init__(model_name, temperature, max_tokens)
        self.expansion_agents, self.evaluator_agent, self.video_generator = self._initialize_agents()
    
    def _initialize_agents(self) -> Tuple[ExpansionAgents, EvaluatorAgent, VideoPromptGenerator]:
        """Initialize the agent pipeline.
        
        Returns:
            Tuple of (ExpansionAgents, EvaluatorAgent, VideoPromptGenerator)
        """
        expansion_agents = ExpansionAgents(self.llm)
        evaluator_agent = EvaluatorAgent(self.llm, expansion_agents)
        video_generator = VideoPromptGenerator(evaluator_agent)
        
        return expansion_agents, evaluator_agent, video_generator
    
    def generate_video_prompt(self, thought: str, stress_level: float) -> str:
        """Generate a video prompt from a thought and stress level.
        
        Uses a three-layer agent system:
        1. Expansion Agents: Add details, steps, and emotions
        2. Evaluator: Synthesize based on stress level
        3. Prompt Generator: Create final video prompt
        
        Args:
            thought: Raw thought from EEG
            stress_level: Current stress level (0-100)
            
        Returns:
            str: Generated video prompt ready for Luma API
        """
        # Layer 1: Expansion Agents
        expansion_results = self.expansion_agents.process_thought(thought)
        
        # Layer 2: Evaluator
        synthesized_description = self.evaluator_agent.evaluate(
            expansion_results,
            stress_level
        )
        
        # Layer 3: Prompt Generator
        prompt = self.video_generator.generate_prompt(
            synthesized_description,
            stress_level
        )
        
        # Format for Luma API
        return self.video_generator.format_for_luma(prompt)

def create_pipeline(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> ThoughtPipeline:
    """Create and initialize the thought-to-video pipeline.
    
    Args:
        model_name: Optional model name override
        temperature: Optional temperature override
        max_tokens: Optional max tokens override
        
    Returns:
        Initialized ThoughtPipeline instance
    """
    return ThoughtPipeline(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )