# QTaskv2

QTaskv2 is an advanced Python toolset for analyzing, querying, and generating C# projects. It leverages AI-powered techniques (Retrieval Augmented Generation, vector databases, LLMs) to provide deep analysis, automated documentation, and code generation for C# codebases.

## Features

- **Automated Indexing**: Parses and indexes C# projects for fast semantic search and analysis.
- **Semantic Query Engine**: Ask high-level questions about the codebase (e.g., “List all API endpoints”) and get context-rich answers.
- **Low Level Design (LLD) Generation**: Automatically generates detailed LLD documents and architecture diagrams (including Mermaid).
- **API Workflow Diagram Generation**: Produces API sequence diagrams and documentation.
- **Project Synthesis**: Generates new C# projects from LLD markdown, including directory structure, code files, and project/solution files.
- **API Server Mode**: Exposes all core features as a REST API for integration into CI/CD or DevOps pipelines.
- **Local-First, Privacy-Respecting**: All processing and data stay local; no external API calls are made by default.

## Directory Structure

```
QTaskv2/
├── main.py                # CLI entry point and orchestrator
├── api_server.py          # REST API server (FastAPI)
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── src/
│   ├── code_parser.py     # C# code parsing logic
│   ├── embeddings.py      # Embedding generation for code
│   ├── vector_store.py    # Vector database management
│   ├── llm_interface.py   # Language model interface
│   ├── rag_pipeline.py    # Retrieval Augmented Generation pipeline
│   ├── diagram_generator.py # LLD and diagram generation
│   └── project_generator.py # C# project generation logic
├── templates/             # Markdown and API documentation templates
├── Output/                # Generated outputs (LLD, code, diagrams)
├── Doc.md                 # Detailed internal documentation
└── Commands.md            # Quick command reference
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Index a C# Project

```bash
python main.py /path/to/your/csharp/project --index
```

### 3. Query the Project

```bash
python main.py /path/to/your/csharp/project --query "How does the authentication system work?"
```

### 4. Generate Low Level Design (LLD)

- For the whole project:
    ```bash
    python main.py /path/to/your/csharp/project --generate-lld
    ```
- For a specific component:
    ```bash
    python main.py /path/to/your/csharp/project --generate-lld "UserService"
    ```

### 5. Generate API Workflow Diagram

```bash
python main.py /path/to/your/csharp/project --generate-api-workflow
```

### 6. Enhance LLD

```bash
python main.py /path/to/your/csharp/project --enhance-lld /path/to/LLD.md
```

### 7. Generate New C# Project from LLD

```bash
python main.py /path/to/your/csharp/project --generate-project /path/to/lld.md /path/to/output
```

### 8. Run as REST API

```bash
python api_server.py
```

Then use `curl` or HTTP clients to POST to endpoints like `/index`, `/query`, `/generate-lld`, `/generate-project`, etc.

## Example Commands

See `Commands.md` for a comprehensive list of CLI commands and sample outputs.

## Important Notes

- **Privacy:** All processing is local. No code or queries are sent externally by default.
- **Persistence:** Indexed vector database persists between sessions. Re-index after major code changes.
- **Performance:** First-time indexing may be slow; subsequent queries are fast.
- **Troubleshooting:** See `Doc.md` for common issues and solutions.

## Contributing

Contributions are welcome! Please see the issues list and open a pull request for improvements or bug fixes.

---

**Project by [0xAdityaSingh](https://github.com/0xAdityaSingh)**
