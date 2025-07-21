#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from src.code_parser import CSharpParser
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.llm_interface import LLMInterface
from src.diagram_generator import DiagramGenerator
from src.project_generator import ProjectGenerator
import logging
import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSharpProjectAnalyzer:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise ValueError(f"Project path {project_path} does not exist")
        
        logger.info("Initializing components...")
        self.parser = CSharpParser()
        self.vector_store = VectorStore()
        self.llm = LLMInterface()
        self.rag = RAGPipeline(self.vector_store, self.llm)
        self.diagram_gen = DiagramGenerator(self.rag)
        self.project_gen = ProjectGenerator(self.rag)
        
    def index_project(self, force_reindex=False):
        """Index the C# project into vector store"""
        logger.info(f"Indexing project: {self.project_path}")
        
        if not force_reindex and self.vector_store.is_indexed(str(self.project_path)):
            logger.info("Project already indexed. Use --force-reindex to reindex.")
            return
        
        # Parse all C# files
        cs_files = list(self.project_path.rglob("*.cs"))
        logger.info(f"Found {len(cs_files)} C# files")
        
        documents = []
        for file in cs_files:
            try:
                doc = self.parser.parse_file(file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error parsing {file}: {e}")
        
        # Index documents
        self.vector_store.index_documents(documents, str(self.project_path))
        logger.info("Indexing complete!")
    
    def query(self, question: str):
        """Query the indexed project"""
        logger.info(f"Query: {question}")
        response = self.rag.query(question, str(self.project_path))
        return response
    
    def generate_lld(self, component: str = None):
        """Generate Low Level Design document"""
        logger.info(f"Generating LLD for: {component or 'entire project'}")
        return self.diagram_gen.generate_lld(str(self.project_path), component)
    
    def generate_api_workflow(self):
        """Generate API workflow diagram"""
        logger.info("Generating API workflow diagram")
        return self.diagram_gen.generate_api_workflow(str(self.project_path))
    
    def generate_project(self, lld_path: str, output_path: str):
        """Generate new C# project from LLD"""
        logger.info(f"Generating project from LLD: {lld_path}")
        self.project_gen.generate_from_lld(
            lld_path, 
            output_path, 
            str(self.project_path)
        )

def simplify_mermaid_generics(content):
    import re
    # Remove generics from class names: class Name[T] -> class Name
    content = re.sub(r'class ([A-Za-z0-9_]+)\[[^\]]*\]', r'class \1', content)
    # Remove generics from type signatures: IList<T> -> IList
    content = re.sub(r'([A-Za-z0-9_]+)<[^>]+>', r'\1', content)
    # Remove any stray [T] or <T> in method/property lines
    content = re.sub(r'\[T\]', '', content)
    content = re.sub(r'<T>', '', content)
    # Remove or replace parentheses and special characters in Mermaid node labels
    def fix_node_label(match):
        label = match.group(1)
        # Replace parentheses with dash, remove other special chars except spaces and alphanum
        label = label.replace('(', ' - ').replace(')', '')
        label = re.sub(r'[^a-zA-Z0-9 \-]', '', label)
        return f'[{label}]'
    content = re.sub(r'\[([^\]]+)\]', fix_node_label, content)
    # Remove generics and square brackets from relationship lines (e.g., ObjectPool[int[]] -> ObjectPool)
    content = re.sub(r'(-->\s*)([A-Za-z0-9_]+)\[[^\]]*\]', r'\1\2', content)
    return content

def save_output(directory, prefix, content, query_or_command=None, is_diagram=False, llm=None):
    from pathlib import Path
    import re
    Path(directory).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix_clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', prefix)[:40]
    filename = f"{prefix_clean}_{timestamp}.md"
    filepath = Path(directory) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        if query_or_command:
            f.write(f"# Query/Command\n\n{query_or_command}\n\n")
        if is_diagram and llm is not None:
            # Simplify generics in Mermaid diagrams
            content_simple = simplify_mermaid_generics(content)
            # Add a note about simplification
            if content != content_simple:
                f.write("> **Note:** Generic type parameters were removed from Mermaid diagrams for compatibility with mermaid.live.\n\n")
            summary_prompt = f"Summarize the following documentation in 1-2 paragraphs, focusing on the high-level architecture, main components, and how the diagrams relate to each other.\n\n{content_simple}"
            summary = llm.generate(summary_prompt)
            f.write(f"# Summary\n\n{summary}\n\n")
            toc = []
            if 'classDiagram' in content_simple or 'mermaid' in content_simple:
                toc.append("- [Diagrams](#diagrams)")
            if 'Appendix' in content_simple:
                toc.append("- [Appendix](#appendix)")
            if toc:
                f.write("# Table of Contents\n\n" + "\n".join(toc) + "\n\n")
            f.write(f"# Details\n\n{content_simple}\n")
        else:
            f.write(f"# Response\n\n{content}\n")
    print(f"\n[Saved output to {filepath}]\n")

def main():
    parser = argparse.ArgumentParser(description='C# Project Analyzer')
    parser.add_argument('project_path', help='Path to C# project')
    parser.add_argument('--index', action='store_true', help='Index the project')
    parser.add_argument('--force-reindex', action='store_true', help='Force reindexing')
    parser.add_argument('--query', help='Query the project')
    parser.add_argument('--generate-lld', nargs='?', const='', help='Generate LLD for component')
    parser.add_argument('--generate-api-workflow', action='store_true', help='Generate API workflow')
    parser.add_argument('--generate-project', nargs=2, metavar=('LLD_PATH', 'OUTPUT_PATH'), 
                       help='Generate project from LLD')
    parser.add_argument('--enhance-lld', metavar='LLD_PATH', help='Enhance LLD markdown using RAG and save as EnhancedLLD.md')
    
    args = parser.parse_args()
    
    try:
        analyzer = CSharpProjectAnalyzer(args.project_path)
        
        if args.index or args.force_reindex:
            analyzer.index_project(args.force_reindex)
        
        if args.query:
            query_prefix = args.query if args.query is not None else ""
            response = analyzer.query(query_prefix)
            print("\nResponse:")
            print(response)
            save_output("Output/Ask", query_prefix, response, query_or_command=query_prefix)
        
        if args.generate_lld is not None:
            component = args.generate_lld if args.generate_lld is not None else ""
            lld = analyzer.generate_lld(component)
            print("\nGenerated LLD:")
            print(lld)
            prefix = f"LLD_{component}" if component else "LLD_project"
            save_output("Output/Diagram", prefix, lld, query_or_command=f"generate-lld {component}", is_diagram=True, llm=analyzer.llm)
        
        if args.generate_api_workflow:
            workflow = analyzer.generate_api_workflow()
            print("\nGenerated API Workflow:")
            print(workflow)
            save_output("Output/Diagram", "API_Workflow", workflow, query_or_command="generate-api-workflow", is_diagram=True, llm=analyzer.llm)
        
        if args.generate_project:
            analyzer.generate_project(args.generate_project[0], args.generate_project[1])
            print(f"\nProject generated at: {args.generate_project[1]}")

        if args.enhance_lld:
            from pathlib import Path
            lld_path = Path(args.enhance_lld)
            if not lld_path.exists():
                print(f"LLD file not found: {lld_path}")
                sys.exit(1)
            lld_text = lld_path.read_text(encoding="utf-8").strip()
            from src.rag_pipeline import RAGPipeline
            from src.vector_store import VectorStore
            from src.llm_interface import LLMInterface
            vector_store = VectorStore()
            llm = LLMInterface()
            global_rag = RAGPipeline(vector_store, llm)
            prompt = f"""
Given the following LLD markdown (which may include a Mermaid class diagram and/or brief notes), generate a comprehensive Low Level Design Document for a C# project in the following format:

# Low Level Design Document

## 1. Component Overview
- **Name**: <Project/Component Name>
- **Purpose**: <Short description of the purpose>
- **Responsibilities**: <Key responsibilities>

## 2. Architecture

### 2.1 Unified Class Diagram
<Mermaid class diagram>

### 2.2 Component List

#### Component: <Name>
- **Type**: <class/interface/etc>
- **File**: <Path> (Output the file path as plain text, with NO Markdown formatting or backticks.)
- **Description**: <Short description>
- **Fields**: <Fields, if any>
- **Constructor**: <Constructor, if any>
- **Methods**: <Methods, if any>

## 3. Data Models
<Description or list>

## 4. Interface Definitions
<Code blocks for interfaces>

## 5. Implementation Details
<Bullet points or paragraphs>

## 6. Error Handling
<Description>

## 7. Performance Considerations
<Description>

## 8. Security Considerations
<Description>

## 9. Testing Strategy
<Description>

---

# Component List (For Code Generator)

For each component, output:
Component: <Name>
Type: <class/interface/etc>
Path: <Path> (Output the file path as plain text, with NO Markdown formatting or backticks.)
Dependencies: <Dependencies>
Fields: <Fields>
Constructor: <Constructor>
Methods: <Methods>

LLD Markdown:
{lld_text}
"""
            rag_response = global_rag.query_global(prompt)
            enhanced_path = lld_path.parent / "EnhancedLLD.md"
            with open(enhanced_path, "w", encoding="utf-8") as f:
                f.write(rag_response)
            # Post-process EnhancedLLD.md to remove backticks from file paths
            import re
            lines = enhanced_path.read_text(encoding="utf-8").splitlines()
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith('Path:') or line.strip().startswith('File:'):
                    # Remove backticks from the path
                    cleaned_lines.append(re.sub(r'`([^`]*)`', r'\1', line))
                else:
                    cleaned_lines.append(line)
            enhanced_path.write_text('\n'.join(cleaned_lines), encoding="utf-8")
            print(f"\nEnhanced LLD saved to: {enhanced_path} (with cleaned file paths)")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 