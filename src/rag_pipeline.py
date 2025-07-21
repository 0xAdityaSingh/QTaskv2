from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store, llm_interface):
        self.vector_store = vector_store
        self.llm = llm_interface
    
    def query(self, question: str, project_path: str) -> str:
        """Process a query using RAG"""
        # Search for relevant chunks
        relevant_chunks = self.vector_store.search(question, project_path, top_k=10)
        
        if not relevant_chunks:
            return "No relevant information found in the codebase."
        
        # Build context
        context = self._build_context(relevant_chunks)
        
        # Generate prompt
        prompt = self._build_prompt(question, context)
        
        system_prompt = """You are a C# code analysis expert. You have deep knowledge of C# programming, 
design patterns, and software architecture. Answer questions based on the provided codebase context.
Be specific and reference actual code when relevant."""
        
        response = self.llm.generate(prompt, system_prompt)
        return response
    
    def query_with_context(self, question: str, project_path: str, additional_context: str) -> str:
        """Query with additional context"""
        relevant_chunks = self.vector_store.search(question, project_path, top_k=8)
        
        context = self._build_context(relevant_chunks)
        context = f"{additional_context}\n\n{context}"
        
        prompt = self._build_prompt(question, context)
        
        system_prompt = """You are a C# code analysis and generation expert. Use the provided context 
to answer questions and generate code that follows the existing patterns and conventions."""
        
        response = self.llm.generate(prompt, system_prompt)
        return response
    
    def query_global(self, question: str, top_k: int = 10) -> str:
        """Process a query using RAG over all indexed projects (global context)"""
        relevant_chunks = self.vector_store.search_all_projects(question, top_k=top_k)
        if not relevant_chunks:
            return "No relevant information found in the codebase."
        context = self._build_context(relevant_chunks)
        prompt = self._build_prompt(question, context)
        system_prompt = """You are a C# code analysis expert. You have deep knowledge of C# programming, \ndesign patterns, and software architecture. Answer questions based on the provided codebase context.\nBe specific and reference actual code when relevant."""
        response = self.llm.generate(prompt, system_prompt)
        return response
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from retrieved chunks"""
        context_parts = []
        
        # Group by type for better organization
        by_type = {}
        for chunk in chunks:
            chunk_type = chunk['type']
            if chunk_type not in by_type:
                by_type[chunk_type] = []
            by_type[chunk_type].append(chunk)
        
        # Build context in order of importance
        for chunk_type in ['file_summary', 'class', 'method']:
            if chunk_type in by_type:
                context_parts.append(f"\n[{chunk_type.upper()} CONTEXT]")
                for chunk in by_type[chunk_type][:3]:  # Limit per type
                    metadata = chunk['metadata']
                    context_parts.append(f"\nFile: {metadata.get('file_path', 'Unknown')}")
                    if 'class_name' in metadata:
                        context_parts.append(f"Class: {metadata['class_name']}")
                    if 'method_name' in metadata:
                        context_parts.append(f"Method: {metadata['method_name']}")
                    context_parts.append(f"\n{chunk['content']}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the final prompt"""
        return f"""Based on the following C# codebase context, answer the question.

CODEBASE CONTEXT:
{context}

QUESTION: {question}

Please provide a detailed and accurate answer based on the code context provided. If you reference specific code, 
mention the file and class/method names.""" 