from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    # def __init__(self, model_name="all-MiniLM-L6-v2"):
    def __init__(self, model_name="all-mpnet-base-v2"):
        print("[INFO] Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)