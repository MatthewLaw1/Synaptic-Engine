"""
This module contains the agent classes used in the video generation pipeline.
Each agent is responsible for a specific task in converting a simple thought
into a detailed video prompt.
"""

import threading
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from .base import BaseLLMPipeline

class ThoughtAgent:
    """Base agent for processing thoughts."""
    
    def __init__(self, llm: BaseLanguageModel, prompt_template: str):
        """Initialize the thought agent.
        
        Args:
            llm: Language model to use
            prompt_template: Template for the prompt
        """
        self.llm = llm
        self.prompt = PromptTemplate.from_template(prompt_template)
        
    def process(self, **kwargs) -> str:
        """Process input using the prompt template and LLM.
        
        Args:
            **kwargs: Arguments for the prompt template
            
        Returns:
            Processed response from the LLM
        """
        chain = self.prompt | self.llm
        return chain.invoke(kwargs).content

class ExpansionAgents:
    """First layer: Expansion agents add details, steps, and emotions."""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize expansion agents with language model.
        
        Args:
            llm: Language model to use
        """
        self.visual_agent = ThoughtAgent(
            llm=llm,
            prompt_template="Enhance this thought with vivid sensory and contextual details:\n{thought}\nFocus on sights, sounds, textures, and environmental context."
        )
        
        self.sequence_agent = ThoughtAgent(
            llm=llm,
            prompt_template="Break this thought into clear, sequential steps:\n{thought}\nFocus on the logical flow and temporal sequence."
        )
        
        self.emotion_agent = ThoughtAgent(
            llm=llm,
            prompt_template="Enhance the emotional elements of this thought:\n{thought}\nFocus on emotional depth, reactions, and interpersonal dynamics."
        )
    
    def process_thought(self, thought: str) -> Dict[str, str]:
        """Process a thought through the agent pipeline.
        
        Args:
            thought: The thought to process
            
        Returns:
            Dict containing the processed results
        """
        return {
            'visual': self.visual_agent.process(thought=thought),
            'sequence': self.sequence_agent.process(thought=thought),
            'emotional': self.emotion_agent.process(thought=thought)
        }

class EvaluatorAgent:
    """Second layer: Synthesizes expansion results based on research-backed 
    stress level correlations."""
    
    def __init__(self, llm: BaseLanguageModel, expansion_agents: ExpansionAgents):
        """Initialize the evaluator agent.
        
        Args:
            llm: Language model to use
            expansion_agents: ExpansionAgents instance
        """
        self.llm = llm
        self.expansion_agents = expansion_agents
        self._latest_result = None
        self._lock = threading.Lock()
    
    def _calculate_stress_weights(self, stress_level: float) -> Dict[str, float]:
        """Calculate weights based on research about stress's effect on cognition.
        
        Research shows:
        - High stress increases attention to emotional stimuli (amygdala activation)
        - High stress narrows visual focus (attentional tunneling)
        - High stress enhances sequential processing for action-readiness
        
        Args:
            stress_level: 0-100 stress level
            
        Returns:
            Dict of weights for each perspective
        """
        # Normalize stress level to 0-1
        stress = stress_level / 100.0
        
        # Research-backed weight calculations
        emotional_weight = 0.2 + (0.4 * stress)  # 0.2-0.6 range
        visual_weight = 0.5 - (0.3 * stress)    # 0.5-0.2 range
        sequence_weight = 0.3 + (0.2 * stress)  # 0.3-0.5 range
        
        # Normalize weights to sum to 1
        total = emotional_weight + visual_weight + sequence_weight
        return {
            'emotional': emotional_weight / total,
            'visual': visual_weight / total,
            'sequence': sequence_weight / total
        }
    
    def evaluate(self, expansion_results: Dict[str, str], stress_level: float) -> str:
        """Evaluate and synthesize the three perspectives based on stress level.
        
        Args:
            expansion_results: Results from the expansion agents
            stress_level: User's stress level (0-100)
        
        Returns:
            Synthesized description
        """
        weights = self._calculate_stress_weights(stress_level)
        
        template = """Synthesize these three perspectives into a coherent scene description, 
        considering their relative importance weights:

        1. Visual Details ({visual_weight:.0%}):
        {visual_details}

        2. Sequential Steps ({sequence_weight:.0%}):
        {sequence_steps}

        3. Emotional Elements ({emotional_weight:.0%}):
        {emotional_elements}

        Stress Level: {stress_level}%

        Guidelines:
        - Give more emphasis to perspectives with higher weights
        - Maintain logical consistency across all perspectives
        - Create a vivid, coherent scene that captures the essence of the thought
        - Keep emotional intensity proportional to the stress level
        - Ensure the description is concrete and filmable

        Synthesized Scene:"""
        
        chain = ThoughtAgent(llm=self.llm, prompt_template=template)
        
        result = chain.process(
            visual_details=expansion_results['visual'],
            sequence_steps=expansion_results['sequence'],
            emotional_elements=expansion_results['emotional'],
            visual_weight=weights['visual'],
            sequence_weight=weights['sequence'],
            emotional_weight=weights['emotional'],
            stress_level=stress_level
        )
        
        return result.strip()

class VideoPromptGenerator(BaseLLMPipeline):
    """Third layer: Converts synthesized description into a structured
    video generation prompt."""
    
    def __init__(
        self,
        evaluator_agent: EvaluatorAgent,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Initialize the video prompt generator.
        
        Args:
            evaluator_agent: EvaluatorAgent instance
            model_name: Optional model name override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
        """
        super().__init__(model_name, temperature, max_tokens)
        self.evaluator_agent = evaluator_agent
    
    def generate_prompt(self, synthesized_description: str, stress_level: float) -> str:
        """Generate a clean, ready-to-use video generation prompt.
        
        Args:
            synthesized_description: Output from EvaluatorAgent
            stress_level: User's stress level (0-100)
            
        Returns:
            Clean prompt ready for video generation service
        """
        template = """Convert this scene into a cinematic video generation prompt:

        Scene: "{description}"
        Stress Level: {stress_level}%

        Format the prompt for optimal video generation:
        - Focus on clear, actionable visual elements
        - Include specific camera movements and angles
        - Specify lighting, atmosphere, and mood
        - Keep it under 100 words
        - Make it compatible with Luma video generation AI
        - Use a cinematic, professional style
        - Avoid abstract concepts, focus on concrete visuals
        - Include camera direction (e.g., close-up, wide shot)
        - Specify time of day and weather if relevant

        The output should be ready to paste directly into Luma AI.

        Video Generation Prompt:"""
        
        prompt_chain = ThoughtAgent(
            llm=self.llm,
            prompt_template=template
        )
        
        return prompt_chain.process(
            description=synthesized_description,
            stress_level=stress_level
        )
    
    def format_for_luma(self, prompt: str) -> str:

        prompt = prompt.strip()
        
        prompt = f"{prompt}, cinematic 8k ultra HD, professional lighting, depth of field"
        
        return prompt