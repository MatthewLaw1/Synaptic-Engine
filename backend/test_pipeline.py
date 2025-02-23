"""
Tests for the agent pipeline.
"""

import unittest
from agent_pipeline import ThoughtPipeline, create_pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test pipeline with test configuration."""
        self.pipeline = create_pipeline(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000
        )
    
    def test_generate_video_prompt(self):
        """Test the complete video prompt generation pipeline."""
        thought = "Walking through a peaceful garden"
        stress_level = 30.0
        
        prompt = self.pipeline.generate_video_prompt(thought, stress_level)
        
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)
        self.assertIn("cinematic", prompt.lower())

if __name__ == '__main__':
    unittest.main()
