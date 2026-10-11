"""
Builds a lightweight TF-IDF search index from the chunks already stored in
chroma_db/ (no re-OCR needed -- the chunk text is read straight out of the
existing index). Produces tfidf_index/{vectorizer.pkl, matrix.npz, meta.json}.

Why TF-IDF instead of the spaCy word-vector embeddings used earlier: spaCy's
en_core_web_md loads a ~685k-word vector table that alone uses ~350MB of RAM,
which doesn't fit in Render's free-tier 512MB limit alongside ChromaDB, Flask
and gunicorn. TF-IDF uses only a few MB and starts instantly.
"""
import json
import pickle
import chromadb
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

CHROMA_DB_FOLDER = "chroma_db"
COLLECTION_NAME = "tuk_mous"
OUT_DIR = "tfidf_index"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    print(f"Loaded {len(ids)} chunks from {CHROMA_DB_FOLDER}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)
    print(f"TF-IDF matrix shape: {matrix.shape}")

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    sp.save_npz(os.path.join(OUT_DIR, "matrix.npz"), matrix)

    meta = [
        {"id": i, "text": d, **m}
        for i, d, m in zip(ids, documents, metadatas)
    ]
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"Saved TF-IDF index to {OUT_DIR}/")


if __name__ == "__main__":
    main()
