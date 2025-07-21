import ollama
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self._ensure_model()
    
    def _ensure_model(self):
        """Ensure embedding model is available"""
        try:
            ollama.show(self.model_name)
        except:
            logger.info(f"Pulling embedding model: {self.model_name}")
            ollama.pull(self.model_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_embedding(text)
                if embedding:
                    batch_embeddings.append(embedding)
                else:
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * 768)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings 