Commands : 

python main.py csharp-sample/csharp --index
# Output:
# Indexing project: csharp-sample/csharp
# Project indexed successfully!

#Query
python main.py csharp-sample/csharp --query "List all API endpoints and their routes in this project."
# Output:
# Query: List all API endpoints and their routes in this project.
# [LLM-generated answer, e.g.]:
# - GET /api/values
# - POST /api/values
# ...

#Diagram Generation
python main.py csharp-sample/csharp --generate-lld
# Output:
# Generating LLD for project: csharp-sample/csharp
# LLD generated at Output/code/ClientServiceSample/LLD.md

python main.py csharp-sample/csharp --generate-api-workflow
# Output:
# Generating API workflow for project: csharp-sample/csharp
# API workflow generated at Output/code/ClientServiceSample/APIWorkflow.md

#LLD Enhancement
python main.py Output/code/ClientServiceSample --enhance-lld Output/code/ClientServiceSample/LLD.md
# Output:
# Enhancing LLD: Output/code/ClientServiceSample/LLD.md
# Enhanced LLD saved to Output/code/ClientServiceSample/EnhancedLLD.md

#Code Generation
python main.py csharp-sample/csharp --generate-project Output/code/ClientServiceSample/LLD.md Output/code/ClientServiceSample
# Output:
# Generating project at Output/code/ClientServiceSample using LLD Output/code/ClientServiceSample/LLD.md
# Project generated successfully!

# Optional: List indexed projects
default: python main.py --list-indexed-projects
# Output:
# Indexed projects:
# - csharp-sample/csharp
# - ...

# Optional: Delete/clear index
default: python main.py csharp-sample/csharp --delete-index
# Output:
# Deleted index for project: csharp-sample/csharp

