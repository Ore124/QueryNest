# Milvus BM25 / Sparse / Hybrid Search POC

## Scope

This POC is isolated from the production retrieval path. It does not change `HybridIndex`, `retrieval.bm25_backend`, or the local BM25 fallback.

Script:

```powershell
cd backend
..\.venv\Scripts\python scripts\poc_milvus_bm25.py
```

## Verified Versions

- Milvus service image: `milvusdb/milvus:v2.6.3`
- PyMilvus runtime: `2.6.16`

Capability check:

```json
{
  "has_function_type_bm25": true,
  "has_hybrid_search": true,
  "has_sparse_float_vector": true,
  "has_rankers": true,
  "supported": true
}
```

## Temporary Collection Schema

The POC creates a temporary collection named like `rag_bm25_poc_<timestamp>` and drops it after the run unless `--keep` is passed.

Fields:

- `chunk_id`: `VARCHAR`, primary key
- `kb_id`: `VARCHAR`
- `doc_id`: `VARCHAR`
- `index_version`: `VARCHAR`
- `text`: `VARCHAR`, `enable_analyzer=true`
- `dense`: `FLOAT_VECTOR`, dim `16`
- `sparse`: `SPARSE_FLOAT_VECTOR`, BM25 function output

Function:

```python
Function(
    name="text_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse"],
)
```

Indexes:

```python
index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
```

## API Calls

Dense search:

```python
client.search(
    collection_name=collection,
    data=[query_dense],
    anns_field="dense",
    filter='kb_id == "kb-poc" and index_version == "poc-v1"',
    search_params={"metric_type": "COSINE"},
)
```

BM25 sparse search:

```python
client.search(
    collection_name=collection,
    data=[query],
    anns_field="sparse",
    filter='kb_id == "kb-poc" and index_version == "poc-v1"',
    search_params={"metric_type": "BM25"},
)
```

Hybrid RRF:

```python
client.hybrid_search(
    collection_name=collection,
    reqs=[dense_req, bm25_req],
    ranker=RRFRanker(k=60),
    limit=5,
)
```

Hybrid weighted:

```python
client.hybrid_search(
    collection_name=collection,
    reqs=[dense_req, bm25_req],
    ranker=WeightedRanker(0.4, 0.6),
    limit=5,
)
```

## Run Result

Query:

```text
API 500 traceId error logs
```

Top result:

```json
{
  "dense_top_chunk": "chunk-003",
  "bm25_top_chunk": "chunk-003",
  "hybrid_rrf_top_chunk": "chunk-003",
  "hybrid_weighted_top_chunk": "chunk-003"
}
```

The winning chunk text was:

```text
For API 500 incidents, check traceId, error logs, and recent deployments.
```

## Recommendation

Milvus BM25 / sparse / hybrid search is supported in the current local stack. The next implementation step should still remain behind `RETRIEVAL_BM25_BACKEND=milvus` and should use a new versioned collection schema or a new collection name, because the current production dense collection schema does not include analyzer-enabled text or sparse BM25 fields.
