import spacy


class SpacyEmbeddingFunction:
    """
    Free, local embedding function using spaCy's en_core_web_md word vectors
    (averaged per document to form one vector per chunk). This is the same
    embedding function used to build the index, so query-time vectors land
    in the same space as the stored chunk vectors.
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def name(self):
        return "spacy-en_core_web_md"

    def __call__(self, input):
        return [self.nlp(text).vector.tolist() for text in input]

    def embed_documents(self, input):
        return self(input)

    def embed_query(self, input):
        return self(input)
