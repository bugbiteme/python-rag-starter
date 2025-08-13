
import sys
import pysqlite3 as sqlite3
sys.modules["sqlite3"] = sqlite3

import base64
import chromadb
import uuid
from ollama_embedding_function import OllamaEmbeddingFunction
from flask import Flask, request, jsonify, Response

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
    Calls the chunks endpoint on another service (http://instructions:8080/chunks)
    and mirrors its response (status code + JSON). Any query params are forwarded.
    """
    remote_url = "http://instructions:8080/chunks"

    try:
        upstream = requests.get(remote_url, params=request.args, timeout=10)
    except requests.RequestException as e:
        return jsonify({
            "error": "failed to reach remote chunks service",
            "details": str(e),
            "remote": remote_url
        }), 502

    # Try to mirror JSON exactly; fall back to raw content if not JSON.
    content_type = upstream.headers.get("Content-Type", "application/json")
    try:
        data = upstream.json()
        return jsonify(data), upstream.status_code
    except ValueError:
        # Not JSON; just relay bytes and content-type
        return Response(upstream.content, status=upstream.status_code, content_type=content_type)


app.run(host="0.0.0.0", port=8080)
