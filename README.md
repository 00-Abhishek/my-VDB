bash '''
MyAI
MyAI is a lightweight vector search and retrieval system built in Python. It combines classical nearest neighbor algorithms with modern embedding-based retrieval, all exposed through a simple web interface and REST API.
The project is designed to be minimal, transparent, and easy to experiment with — no heavy dependencies, no hidden abstractions.

Overview
MyAI provides:


Multiple vector search strategies (exact + approximate)


Interactive browser-based visualization


Local document ingestion and retrieval (RAG)


A clean, dependency-light Python backend


It is suitable for learning, experimentation, and small-scale local deployments.

Features
AreaDetailsSearch algorithmsHNSW, KD-Tree, Brute ForceDistance metricsCosine, Euclidean, ManhattanVector dataPreloaded 16D vectors across multiple domainsVisualizationPCA-based scatter plot in browserDocument pipelineChunk → Embed → Store → RetrieveRAGRetrieval + generation via local LLMAPIREST endpoints for all core operations

Tech Stack


Backend: Python (standard library only)


Frontend: Vanilla HTML + JS


Embeddings & LLM: Ollama (local)



Requirements


Python 3.11+


(Optional) Ollama for embeddings and generation


No external Python packages are required.

Quick Start
1. Verify Python
python --version

2. Install Ollama (optional but recommended)
Download from: https://ollama.com
Pull required models:
ollama pull nomic-embed-textollama pull llama3.2

3. Run the application
cd myaipython app.py
Server starts at:
http://localhost:8080

4. Open in browser
http://localhost:8080

Project Structure
myai/├── app.py├── index.html├── README.md├── requirements.txt└── docs/    ├── API.md    └── ARCHITECTURE.md

Architecture


app.py
Hosts the HTTP server and all API endpoints


VectorDB
Handles in-memory vectors and search algorithms


DocumentDB
Stores embedded document chunks for retrieval


OllamaClient
Interfaces with local Ollama models


index.html
Frontend UI + visualization logic



API Endpoints
MethodEndpointDescriptionGET/searchQuery vector searchPOST/insertInsert vectorDELETE/delete/:idDelete vectorGET/itemsList vectorsGET/benchmarkCompare algorithmsGET/hnsw-infoInspect HNSW structurePOST/doc/insertAdd documentGET/doc/listList document chunksDELETE/doc/delete/:idDelete chunkPOST/doc/searchRetrieve chunksPOST/doc/askRAG queryGET/statusSystem status
Detailed specs: docs/API.md

Example Usage
Vector search
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

Ask a question (RAG)
curl -X POST http://localhost:8080/doc/ask \  -H "Content-Type: application/json" \  -d '{"question":"What is dynamic programming?","k":3}'

Behavior Notes


Core vector search works independently of Ollama


Document ingestion and RAG require Ollama running locally


Frontend and backend run on the same origin (no separate server needed)



Troubleshooting
IssueResolutionPython not foundInstall Python 3.11+ and add to PATHPort 8080 in useChange port in app.pyOllama not respondingRun ollama serveModels missingPull required modelsDocs not workingCheck Ollama status

Documentation


API: docs/API.md


Architecture: docs/ARCHITECTURE.md



Positioning (important for authenticity)
This project intentionally avoids heavy frameworks to:


Keep system behavior transparent


Make algorithmic comparisons easy


Enable local-first experimentation


It is not intended as a production-scale vector database, but as a clear and extensible reference implementation.
'''
