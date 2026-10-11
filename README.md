# TUK MOU Search

A small search tool over scanned MOU documents for the TUK Directorate of Community
Outreach, Linkages and Partnerships. Built as part of a pilot RAG (retrieval) project.

## How it works

- `chroma_db/` holds a prebuilt ChromaDB index (158 chunks from 7 MOU PDFs), created by
  OCR'ing scanned documents, splitting them into section-level chunks, and embedding
  each chunk.
- Embeddings use spaCy's `en_core_web_md` averaged word vectors (a free, local
  alternative used because the original build environment couldn't reach Hugging Face
  to download a proper sentence-transformer model). Retrieval quality is decent for
  close keyword matches, weaker on paraphrased/indirect queries -- a known limitation
  worth upgrading later.
- `app.py` is a small Flask app that loads the index once at startup and serves a
  single search page.

## Running locally

```
pip install -r requirements.txt
python app.py
```

## Adding more documents

The indexing pipeline (OCR, chunking, embedding) that built `chroma_db/` lives in a
separate project directory, not in this repo. To add documents, re-run that pipeline
and replace `chroma_db/` here, then redeploy.
