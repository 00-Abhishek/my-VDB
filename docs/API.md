# MyAI API Reference

All endpoints are served from:

```text
http://localhost:8080
```

## Demo vector endpoints

### `GET /search`

Searches the 16D demo vector database.

Query parameters:

- `v`: comma-separated 16D vector
- `k`: number of results, default `5`
- `metric`: `cosine`, `euclidean`, or `manhattan`
- `algo`: `hnsw`, `kdtree`, or `bruteforce`

Example:

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### `POST /insert`

Inserts a new demo vector.

Body:

```json
{
  "metadata": "Binary heap operations",
  "category": "cs",
  "embedding": [0.9, 0.8, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
}
```

### `DELETE /delete/:id`

Deletes a demo vector by id.

Example:

```powershell
curl -X DELETE http://localhost:8080/delete/3
```

### `GET /items`

Returns all demo vectors.

### `GET /benchmark`

Benchmarks brute force, KD-Tree, and HNSW on the same query.

Query parameters:

- `v`: comma-separated 16D vector
- `k`: number of results, default `5`
- `metric`: `cosine`, `euclidean`, or `manhattan`

### `GET /hnsw-info`

Returns HNSW graph metadata, including:

- top layer
- node count
- nodes per layer
- edges per layer

### `GET /stats`

Returns demo database metadata:

- item count
- dimensions
- available algorithms
- available metrics

## Document and RAG endpoints

### `POST /doc/insert`

Splits a document into chunks, embeds each chunk with Ollama, and stores the results.

Body:

```json
{
  "title": "Operating Systems Notes",
  "text": "Long document text goes here..."
}
```

Response includes:

- inserted chunk ids
- chunk count
- embedding dimension

### `GET /doc/list`

Returns stored document chunks with:

- `id`
- `title`
- `preview`
- `words`

### `DELETE /doc/delete/:id`

Deletes a stored document chunk.

### `POST /doc/search`

Retrieves the nearest document chunks without generating an answer.

Body:

```json
{
  "question": "What is dynamic programming?",
  "k": 3
}
```

### `POST /doc/ask`

Runs the full retrieval plus generation flow.

Body:

```json
{
  "question": "What is dynamic programming?",
  "k": 3
}
```

Response includes:

- generated answer
- generation model
- retrieved chunks
- current document count

### `GET /status`

Returns the current runtime status:

- whether Ollama is available
- embed model name
- generation model name
- demo vector count and dimensions
- document chunk count and dimensions

## Notes

- Demo search endpoints work without Ollama.
- Document endpoints depend on a local Ollama server at `127.0.0.1:11434`.
- The server also responds to `OPTIONS` for browser CORS requests.
