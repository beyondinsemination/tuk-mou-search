import os
import chromadb
from flask import Flask, render_template, request

from embeddings import SpacyEmbeddingFunction

DB_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "tuk_mous"

app = Flask(__name__)

# Loaded once at startup, reused across requests.
_embedding_fn = SpacyEmbeddingFunction()
_client = chromadb.PersistentClient(path=DB_FOLDER)
_collection = _client.get_collection(COLLECTION_NAME, embedding_function=_embedding_fn)


def run_search(query, n_results=8):
    if not query or not query.strip():
        return []
    results = _collection.query(query_texts=[query], n_results=n_results)
    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "source_file": meta.get("source_file", "unknown"),
            "chunk_index": meta.get("chunk_index", "?"),
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("q", "").strip()
    results = run_search(query) if query else []
    return render_template(
        "index.html",
        query=query,
        results=results,
        total_chunks=_collection.count(),
    )


@app.route("/healthz")
def healthz():
    return {"status": "ok", "chunks": _collection.count()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
