# import numpy as np

# class HybridRetriever:
#     def __init__(self, bm25, vector_store, embedder):
#         self.bm25 = bm25
#         self.vector_store = vector_store
#         self.embedder = embedder

#     def search(self, query, k=5, alpha=0.5, beta=0.5):
#         """Hybrid search combining BM25 and vector similarity"""
#         bm25_results = self.bm25.search(query, k=k)
#         vec_results = self.vector_store.search(query, self.embedder, k=k)

#         # merge results with weights
#         merged = {}
#         for chunk in set(bm25_results + vec_results):
#             bm25_score = 1.0 if chunk in bm25_results else 0.0
#             vec_vector = self.vector_store.get_vector(chunk)
#             query_vector = self.embedder.encode([query])[0]
#             vec_score = cosine_similarity([vec_vector], [query_vector])[0][0]
#             final_score = alpha * bm25_score + beta * vec_score
#             merged[chunk] = final_score

#         # sort by final score
#         sorted_chunks = sorted(merged, key=lambda x: merged[x], reverse=True)
#         return sorted_chunks[:k]

#     def rerank(self, query, chunks, llm):
#         """Optional: Rerank chunks based on LLM scoring"""
#         # Here we can keep simple length or keyword scoring for now
#         scores = []
#         for chunk in chunks:
#             text = chunk['content'] if isinstance(chunk, dict) else str(chunk)
#             # simplistic score: length of text (you can expand)
#             score = len(text)
#             scores.append(score)
#         sorted_chunks = [c for _, c in sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)]
#         return sorted_chunks

#BESTTT
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        # self.index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
        self.chunks = []

    def add(self, vectors, chunks):
        # vectors = np.array(vectors).astype("float32")
        # self.index.add(vectors)
        vectors = np.array(vectors).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

        print(f"[INFO] Added {len(chunks)} chunks to FAISS")

    def search(self, query, embedder, k=2):
        # Convert query → embedding
        query_vector = embedder.encode([query])

        # Convert to float32 (FAISS requirement)
        # query_vector = np.array(query_vector).astype("float32")
        query_vector = np.array(query_vector).astype("float32")

        # Normalize query vector
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        # results = [self.texts[i] for i in indices[0]]
        results = [self.chunks[i] for i in indices[0]]
        return results