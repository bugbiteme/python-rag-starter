import base64
import chromadb
import uuid
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from flask import Flask, request

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

app.run(host="0.0.0.0", port=8080)
