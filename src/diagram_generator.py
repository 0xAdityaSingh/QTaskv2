import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DiagramGenerator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def generate_lld(self, project_path: str, component: Optional[str] = None) -> str:
        """Generate Low Level Design document"""
        if component:
            question = f"""Generate a detailed Low Level Design (LLD) document for the {component} component.
            Include:
            1. Component Overview
            2. A single, comprehensive Mermaid class diagram at the top, showing all classes, interfaces, and their relationships for this component (do not split into multiple diagrams)
            3. Sequence Diagrams for key operations
            4. Data Models
            5. Interface Definitions
            6. Dependencies and Interactions
            7. Error Handling Strategy
            8. Performance Considerations"""
        else:
            question = """Generate a comprehensive Low Level Design (LLD) document for the entire project.
            At the very top, include a single, unified Mermaid class diagram (classDiagram) showing all major classes, interfaces, and their relationships across the project (do not split into multiple diagrams). Then include:
            1. System Architecture Overview
            2. Component Breakdown
            3. Data Flow Diagrams
            4. Database Schema (if applicable)
            5. API Specifications
            6. Security Architecture
            7. Deployment Architecture"""
        
        response = self.rag.query(question, project_path)
        
        # Format as markdown document
        lld = f"""# Low Level Design Document
{'## Component: ' + component if component else '## System Design'}

Generated from codebase analysis.

---

{response}

---

## Appendix

### Mermaid Diagram Rendering
To view the diagrams, use any Mermaid-compatible viewer or paste into:
- https://mermaid.live/
- VS Code with Mermaid extension
- Any markdown viewer with Mermaid support
"""
        
        return lld
    
    def generate_api_workflow(self, project_path: str) -> str:
        """Generate API workflow diagram"""
        question = """Analyze all API endpoints and controllers in the project and generate:
        1. Complete API endpoint inventory
        2. Request/Response flow diagrams (in Mermaid format)
        3. Authentication and authorization flow
        4. API workflow sequence diagrams for major use cases
        5. Error handling patterns
        6. Integration points with external services
        
        Focus on creating clear Mermaid diagrams that show the flow of API requests through the system."""
        
        response = self.rag.query(question, project_path)
        
        workflow = f"""# API Workflow Documentation

Generated from codebase analysis.

---

{response}

---

## API Testing Guide

Based on the above workflows, here are curl examples for testing:

```bash
# Add your API testing examples here
```
"""
        
        return workflow
    
    def generate_class_diagram(self, project_path: str, namespace: str = None) -> str:
        """Generate class diagram for specific namespace or entire project"""
        if namespace:
            question = f"Generate a detailed Mermaid class diagram for the {namespace} namespace showing all classes, interfaces, properties, methods, and relationships."
        else:
            question = "Generate a comprehensive Mermaid class diagram showing the main classes, interfaces, and their relationships across the project."
        
        response = self.rag.query(question, project_path)
        return response 