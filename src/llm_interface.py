import ollama
import config
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self._ensure_model()
    
    def _ensure_model(self):
        """Ensure LLM model is available"""
        try:
            ollama.show(self.model_name)
            logger.info(f"Model {self.model_name} is available")
        except:
            logger.info(f"Pulling model: {self.model_name}")
            ollama.pull(self.model_name)
    
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = None) -> str:
        """Generate response from LLM"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': temperature or config.TEMPERATURE,
                    'num_predict': config.MAX_TOKENS,
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def generate_streaming(self, prompt: str, system_prompt: str = None):
        """Generate streaming response from LLM"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    'temperature': config.TEMPERATURE,
                    'num_predict': config.MAX_TOKENS,
                }
            )
            
            for chunk in stream:
                yield chunk['message']['content']
                
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error: {str(e)}" 