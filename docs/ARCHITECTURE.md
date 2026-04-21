# MyAI Architecture

MyAI is a Python rewrite of the original demo with the same overall learning goals:

- compare multiple nearest-neighbor search strategies
- visualize semantic proximity in the browser
- show how local retrieval and generation can be layered on top

## Backend layout

The backend lives in [app.py](../app.py) and is split into a few clear pieces.

### `VectorDB`

`VectorDB` stores the 16D demo vectors and exposes:

- insert
- delete
- list
- search
- benchmark
- HNSW graph inspection

It keeps three indexes in sync:

- `BruteForce`
- `KDTree`
- `HNSW`

### `DocumentDB`

`DocumentDB` stores document chunks and their real Ollama embeddings.

- For small collections it can fall back to brute force.
- For larger collections it uses HNSW with cosine distance.
- Retrieval is filtered by a max cosine distance threshold.

### `OllamaClient`

`OllamaClient` talks directly to the local Ollama HTTP API with the Python standard library.

Endpoints used:

- `GET /api/tags`
- `POST /api/embeddings`
- `POST /api/generate`

### HTTP server

The app uses `ThreadingHTTPServer` from the Python standard library.

Routes are handled by `MyAIHandler`, which serves:

- the frontend HTML
- JSON API endpoints
- document and RAG requests

## Search algorithms

### Brute force

- exact search
- simplest baseline
- linear scan over all vectors

### KD-Tree

- exact search for the 16D demo vectors
- recursively splits by axis
- useful for showing why classic tree search gets weaker as dimensions rise

### HNSW

- approximate nearest-neighbor graph
- multilayer structure with sparse upper layers
- used both for demo vectors and document retrieval

The Python implementation mirrors the original structure closely:

- random level assignment
- top-down entry-point descent
- layer-local candidate search
- bounded neighbor selection

## Frontend layout

The frontend is in [index.html](../index.html).

It includes:

- a minimal light-themed layout
- a PCA scatter plot for the 16D demo vectors
- controls for algorithm, metric, and top-k
- document insertion and document list views
- an Ask AI panel for retrieval plus generation

The browser also creates lightweight 16D proxy vectors from text so search and RAG actions can be shown on the demo map.

## Data flow

### Demo search flow

1. Browser text query
2. Browser-side 16D keyword embedding
3. `GET /search`
4. Python backend search
5. Results rendered in cards and on the PCA map

### Document insert flow

1. Browser sends title and text
2. Backend chunks the document
3. Backend calls Ollama embeddings
4. Chunks are inserted into `DocumentDB`
5. Browser refreshes the list and visual state

### Ask AI flow

1. Browser sends question to `/doc/search` for visualization context
2. Browser sends question to `/doc/ask`
3. Backend embeds the question
4. Backend retrieves top chunks
5. Backend builds a prompt
6. Backend calls Ollama generation
7. Browser renders the answer and the retrieved chunk details

## Why this version is easier to work with

- Python is simpler to run and modify than the original C++ build.
- No compile step is required.
- The API and behavior stay close to the original project.
- The UI is more minimal while preserving the educational features.
