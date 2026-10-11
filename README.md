# TUK MOU Search

A small search tool over scanned MOU documents for the TUK Directorate of Community
Outreach, Linkages and Partnerships. Built as part of a pilot RAG (retrieval) project.

## How it works

- `chroma_db/` holds the original chunk-level index (158 chunks from 7 MOU PDFs),
  built by OCR'ing scanned documents and splitting them into section-level chunks.
  It's kept here as the source of truth for the chunk text and metadata.
- `tfidf_index/` is what the deployed app actually searches: a TF-IDF (keyword-weighted)
  index built from the same 158 chunks via `build_tfidf_index.py`. It replaced an
  earlier attempt that used spaCy word-vector embeddings -- those needed ~350MB of RAM
  just to load the vector table, which didn't fit Render's free-tier 512MB limit and
  caused the app to be OOM-killed on every deploy. TF-IDF uses a few MB, starts
  instantly, and in testing retrieved exact-term queries (e.g. "contact details for
  notices") *better* than the word-vector approach, which missed them.
- `app.py` is a small Flask app that loads the TF-IDF index once at startup and serves
  a single search page.

## Running locally

```
pip install -r requirements.txt
python app.py
```

## Rebuilding the index after adding documents

The OCR/chunking pipeline that built `chroma_db/` lives in a separate project
directory, not in this repo. After adding new chunks to `chroma_db/`, rebuild the
TF-IDF index the deployed app actually uses:

```
pip install -r requirements.txt -r requirements-build.txt
python build_tfidf_index.py
```

Then commit and push the updated `tfidf_index/` folder to redeploy.
