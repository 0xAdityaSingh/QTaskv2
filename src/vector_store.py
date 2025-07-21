import lancedb
from pathlib import Path
import json
from typing import List, Dict, Any
import logging
from src.embeddings import EmbeddingGenerator
import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.db = lancedb.connect(str(config.VECTOR_DB_PATH))
        self.embedding_gen = EmbeddingGenerator()
        
    def is_indexed(self, project_path: str) -> bool:
        """Check if a project is already indexed"""
        try:
            table_name = self._get_table_name(project_path)
            self.db.open_table(table_name)
            return True
        except:
            return False
    
    def index_documents(self, documents: List[Dict], project_path: str):
        """Index documents into vector store"""
        import pyarrow as pa
        table_name = self._get_table_name(project_path)
        
        # Prepare data for indexing
        records = []
        all_chunks = []
        
        for doc in documents:
            for chunk in doc['chunks']:
                all_chunks.append(chunk)
        
        logger.info(f"Indexing {len(all_chunks)} chunks...")
        
        # Generate embeddings in batches
        texts = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedding_gen.generate_embeddings_batch(texts)
        
        # Ensure all embeddings are the same length
        dim = len(embeddings[0]) if embeddings and embeddings[0] else 768
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            if not embedding or len(embedding) != dim:
                embedding = [0.0] * dim
            record = {
                'id': i,
                'content': chunk['content'],
                'embedding': embedding,
                'type': chunk['type'],
                'metadata': json.dumps(chunk['metadata'])
            }
            records.append(record)
        
        # Create or replace table with fixed-size vector schema
        logger.info(f"Creating table: {table_name}")
        if self.is_indexed(project_path):
            self.db.drop_table(table_name)
        
        schema = pa.schema([
            pa.field('id', pa.int64()),
            pa.field('content', pa.string()),
            pa.field('embedding', pa.list_(pa.float64(), dim)),
            pa.field('type', pa.string()),
            pa.field('metadata', pa.string()),
        ])
        
        self.db.create_table(table_name, records, schema=schema)
        logger.info(f"Indexed {len(records)} chunks successfully")
    
    def search(self, query: str, project_path: str, top_k: int = 10) -> List[Dict]:
        """Search for relevant chunks"""
        table_name = self._get_table_name(project_path)
        
        try:
            table = self.db.open_table(table_name)
            
            # Generate query embedding
            query_embedding = self.embedding_gen.generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search (explicitly specify vector column)
            results = table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_list()
            
            # Parse results
            parsed_results = []
            for result in results:
                parsed_results.append({
                    'content': result['content'],
                    'type': result['type'],
                    'metadata': json.loads(result['metadata']),
                    'score': result.get('_distance', 0)
                })
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def search_all_projects(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for relevant chunks across all indexed projects"""
        table_names = self.db.table_names()
        query_embedding = self.embedding_gen.generate_embedding(query)
        all_results = []
        for table_name in table_names:
            try:
                table = self.db.open_table(table_name)
                results = table.search(query_embedding, vector_column_name="embedding").limit(top_k).to_list()
                for result in results:
                    all_results.append({
                        'content': result['content'],
                        'type': result['type'],
                        'metadata': json.loads(result['metadata']),
                        'score': result.get('_distance', 0)
                    })
            except Exception as e:
                logger.error(f"Error searching table {table_name}: {e}")
        all_results.sort(key=lambda x: x['score'])
        return all_results[:top_k]
    
    def _get_table_name(self, project_path: str) -> str:
        """Generate table name from project path"""
        # Simple hash of the path
        import hashlib
        return f"project_{hashlib.md5(project_path.encode()).hexdigest()[:8]}" 