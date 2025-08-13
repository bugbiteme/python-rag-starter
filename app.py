
import sys
import pysqlite3 as sqlite3
sys.modules["sqlite3"] = sqlite3

import base64
import chromadb
import uuid
import requests  # <-- add this

from ollama_embedding_function import OllamaEmbeddingFunction
from flask import Flask, request, jsonify, Response  # Response already here; keep this one

app = Flask(__name__)

client = chromadb.HttpClient(host="chroma", port=8080)

# Create a collection within the vector DB.
# This is a mandatory step.

collection = client.create_collection("my_collection",
                                      embedding_function=OllamaEmbeddingFunction(url="http://ollama-embedding:8080/api/embeddings", model_name="nomic-embed-text"),
                                      get_or_create=True)

# Let's add some text to the vector DB!
# `documents` is an array of text chunks
# `ids` is an array of unique IDs for each document.
# For our simple purposes, we can use random UUIDs as the `ids`.

text = 'In space, nobody can hear you scream because space is a vacuum, meaning it has no air or other medium to carry sound waves. Sound waves require a medium like air, water, or solid objects to travel through. In the vacuum of space, there is no such medium, so sound cannot propagate. As a result, any sound you make, including a scream, would not be heard by anyone else.'

collection.add(documents=[text], ids=[str(uuid.uuid4())])

# Now let's query the DB!
# We'll just print out the result to the console so we can see that the
# database and embedding model are working.

# `query_texts` is an array of query text.
# `n_results` is the number of closest results chromadb should return.

print(collection.query(query_texts=["Why can nobody hear you scream in space?"],
                       n_results=1))

@app.route('/')
def hello():
    return "Hello. This is a sample application."

from flask import Response  # add this import

@app.route("/remote-chunks", methods=["GET"])
def remote_chunks():
    """
    Calls the chunks endpoint on another service (http://turtlecomms:8080/chunks)
    and mirrors its response (status code + JSON). Any query params are forwarded.
    """
    remote_url = "http://turtlecomms:8080/chunks"

    try:
        upstream = requests.get(remote_url, params=request.args, timeout=10)
    except requests.RequestException as e:
        return jsonify({
            "error": "failed to reach remote chunks service",
            "details": str(e),
            "remote": remote_url
        }), 502

    content_type = upstream.headers.get("Content-Type", "application/json")
    try:
        return jsonify(upstream.json()), upstream.status_code
    except ValueError:
        return Response(upstream.content, status=upstream.status_code, content_type=content_type)

@app.route("/populate", methods=["GET", "POST"])
def populate():
    """
    Populate ChromaDB from internal chunks service.
    Sources chunks from: http://turtlecomms:8080/chunks
    Accepts optional ?size=&url= (also allowed in POST JSON).

    Idempotent: deterministic UUIDv5 per (source_url, chunk_index).
    Uses collection.upsert() if available; otherwise add() with update fallback.
    """
    # Collect optional params (query takes precedence over JSON)
    url_arg = request.args.get("url")
    size_arg = request.args.get("size")
    body = request.get_json(silent=True) or {} if request.method == "POST" else {}

    params = {}
    if url_arg or body.get("url"):
        params["url"] = url_arg or body.get("url")
    if size_arg or body.get("size"):
        params["size"] = str(size_arg or body.get("size"))

    # Call internal service directly
    remote_url = "http://turtlecomms:8080/chunks"
    try:
        r = requests.get(remote_url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except requests.Timeout as e:
        return jsonify({"error": "timeout calling turtlecomms", "details": str(e)}), 504
    except requests.RequestException as e:
        return jsonify({"error": "failed to call turtlecomms", "details": str(e)}), 502
    except ValueError:
        return jsonify({"error": "turtlecomms did not return JSON"}), 502

    chunks = payload.get("chunks", [])
    source_url = payload.get("url") or params.get("url") or "unknown"

    if not chunks:
        return jsonify({
            "added": 0,
            "chunks_count": payload.get("chunks_count", 0),
            "source_url": source_url,
            "message": "No chunks to add."
        }), 200

    # Prepare docs
    documents, ids, metadatas = [], [], []
    for ch in chunks:
        idx = ch.get("index")
        paragraphs = ch.get("paragraphs") or []
        if not paragraphs:
            continue

        doc = "\n\n".join(paragraphs)
        stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url}::chunk::{idx}"))

        documents.append(doc)
        ids.append(stable_id)
        metadatas.append({
            "source_url": source_url,
            "chunk_index": idx,
            "start_paragraph": ch.get("start_paragraph"),
            "end_paragraph": ch.get("end_paragraph"),
            "size": ch.get("size"),
            "chunk_size": payload.get("chunk_size"),
            "paragraph_count_total": payload.get("paragraph_count"),
        })

    # Upsert/add into Chroma
    try:
        if hasattr(collection, "upsert"):
            collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
        else:
            try:
                collection.add(documents=documents, ids=ids, metadatas=metadatas)
            except Exception:
                # Fallback: attempt update for already-present IDs
                if hasattr(collection, "update"):
                    collection.update(ids=ids, documents=documents, metadatas=metadatas)
                else:
                    raise
    except Exception as e:
        return jsonify({"error": "failed to write to Chroma", "details": str(e)}), 500

    return jsonify({
        "added": len(documents),
        "chunks_count": payload.get("chunks_count"),
        "source_url": source_url,
        "collection": getattr(collection, "name", "my_collection"),
        "ids_sample": ids[:5]
    }), 200

@app.route("/stats", methods=["GET"])
def stats():
    """
    Returns a quick count of documents in the ChromaDB collection
    and a small sample (id, preview, metadata).
    """
    try:
        total_docs = collection.count()
    except Exception as e:
        return jsonify({"error": "count failed", "details": str(e)}), 500

    sample = []
    try:
        # In chromadb 0.5.x, valid include items are: "documents", "metadatas", "embeddings", "uris", "data"
        res = collection.get(limit=3, include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        for id_, doc, meta in zip(ids, docs, metas):
            sample.append({
                "id": id_,
                "doc_preview": (doc[:160] + "…") if doc and len(doc) > 160 else doc,
                "metadata": meta or {}
            })
    except Exception as e:
        sample = [{"warning": f"failed to retrieve sample: {e}"}]

    return jsonify({
        "collection": getattr(collection, "name", "my_collection"),
        "total_docs": total_docs,
        "sample": sample
    }), 200
app.run(host="0.0.0.0", port=8080)
