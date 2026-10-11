import os
import json
import pickle
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfidf_index")

app = Flask(__name__)

# Loaded once at startup, reused across requests -- all lightweight (a few MB).
with open(os.path.join(INDEX_DIR, "vectorizer.pkl"), "rb") as f:
    _vectorizer = pickle.load(f)
_matrix = sp.load_npz(os.path.join(INDEX_DIR, "matrix.npz"))
with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
    _meta = json.load(f)


def run_search(query, n_results=8):
    if not query or not query.strip():
        return []
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _matrix).flatten()
    top_idx = scores.argsort()[::-1][:n_results]
    hits = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        m = _meta[i]
        hits.append({
            "source_file": m.get("source_file", "unknown"),
            "chunk_index": m.get("chunk_index", "?"),
            "text": m.get("text", ""),
            "score": float(scores[i]),
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
        total_chunks=len(_meta),
    )


@app.route("/healthz")
def healthz():
    return {"status": "ok", "chunks": len(_meta)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
