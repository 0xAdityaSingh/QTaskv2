from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import CSharpProjectAnalyzer
import uvicorn
from src.lld_mermaid_parser import extract_components_from_lld
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from pathlib import Path
import traceback

app = FastAPI(title="C# Project Analyzer API")

# Store analyzer instances
analyzers = {}

# Initialize RAG pipeline for global context
vector_store = VectorStore()
llm = LLMInterface()
global_rag = RAGPipeline(vector_store, llm)

class ProjectPath(BaseModel):
    path: str

class Query(BaseModel):
    question: str
    project_path: str

class LLDRequest(BaseModel):
    project_path: str
    component: str = None

class ProjectGenRequest(BaseModel):
    project_path: str
    lld_path: str
    output_path: str

class LLDTextRequest(BaseModel):
    lld_text: str

class EnhanceLLDRequest(BaseModel):
    lld_path: str

@app.post("/index")
async def index_project(project: ProjectPath):
    try:
        analyzer = CSharpProjectAnalyzer(project.path)
        analyzer.index_project()
        analyzers[project.path] = analyzer
        return {"status": "success", "message": "Project indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_project(query: Query):
    if query.project_path not in analyzers:
        analyzer = CSharpProjectAnalyzer(query.project_path)
        analyzers[query.project_path] = analyzer
    
    response = analyzers[query.project_path].query(query.question)
    return {"response": response}

@app.post("/generate-lld")
async def generate_lld(request: LLDRequest):
    if request.project_path not in analyzers:
        analyzer = CSharpProjectAnalyzer(request.project_path)
        analyzers[request.project_path] = analyzer
    
    lld = analyzers[request.project_path].generate_lld(request.component)
    return {"lld": lld}

@app.post("/generate-api-workflow")
async def generate_api_workflow(project: ProjectPath):
    if project.path not in analyzers:
        analyzer = CSharpProjectAnalyzer(project.path)
        analyzers[project.path] = analyzer
    
    workflow = analyzers[project.path].generate_api_workflow()
    return {"workflow": workflow}

@app.post("/generate-project")
async def generate_project(request: ProjectGenRequest):
    if request.project_path not in analyzers:
        analyzer = CSharpProjectAnalyzer(request.project_path)
        analyzers[request.project_path] = analyzer
    
    analyzer.generate_project(request.lld_path, request.output_path)
    return {"status": "success", "output_path": request.output_path}

@app.post("/enhance-lld")
def enhance_lld(request: EnhanceLLDRequest):
    """
    Accepts a path to an LLD markdown file, uses RAG (all indexed projects) to generate a comprehensive, production-quality Low Level Design Document in the specified format, saves the result as EnhancedLLD.md in the same directory, and returns the enhanced markdown.
    """
    try:
        # Read the LLD markdown from the provided file path
        lld_path = Path(request.lld_path)
        if not lld_path.exists():
            raise HTTPException(status_code=400, detail=f"LLD file not found: {lld_path}")
        lld_text = lld_path.read_text(encoding="utf-8").strip()

        # Prompt for the LLM
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
        # Save to EnhancedLLD.md in the same directory as the input file
        enhanced_path = lld_path.parent / "EnhancedLLD.md"
        with open(enhanced_path, "w", encoding="utf-8") as f:
            f.write(rag_response)
        return {"enhanced_lld": rag_response, "output_path": str(enhanced_path)}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ENHANCE_LLD ERROR] {tb}")
        return {"error": str(e), "traceback": tb}

@app.post("/enhanced-lld")
def enhanced_lld(request: EnhanceLLDRequest):
    return enhance_lld(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 