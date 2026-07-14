# modules/rag/hybrid_retriever.py

import numpy as np
import faiss
import re

def safe_normalize(scores):
    import numpy as np

    if np.max(scores) - np.min(scores) < 1e-6:
        return np.zeros_like(scores)

    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

class HybridRetriever:
    def __init__(self, bm25, vector_store, embedder):
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder

    
    # ---------------------------
    # SEARCH
    # ---------------------------
    def search(self, query, k=3, alpha=0.5, beta=0.5):

        # BM25 (FIXED)
        tokenized_query = self.bm25.preprocess(query)
        bm25_scores = self.bm25.bm25.get_scores(tokenized_query)

        # FAISS
        query_vector = self.embedder.encode([query])[0]
        query_vector = np.array([query_vector]).astype("float32")
        faiss.normalize_L2(query_vector)

        # Reduced semantic dominance
        if beta > alpha:
            # semantic-heavy (adaptive case)
            faiss_k = max(k * 3, 10)
        else:
            # fixed / keyword-heavy
            faiss_k = max(k * 5, 20)

        # faiss_k = max(k * 8, 30)

        distances, indices = self.vector_store.index.search(
            query_vector,
            faiss_k
        )

        faiss_scores_full = np.zeros(len(self.vector_store.chunks))

        for idx, score in zip(indices[0], distances[0]):
            faiss_scores_full[idx] = score

        faiss_scores = faiss_scores_full

        # Normalize
        # bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
        # faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores) + 1e-8)

        bm25_scores = safe_normalize(bm25_scores)
        faiss_scores = safe_normalize(faiss_scores)

        # Combine
        # final_scores = alpha * bm25_scores + (beta*1.2) * faiss_scores
        
        if beta > alpha:
        # adaptive semantic boost
            final_scores = alpha * bm25_scores + (beta * 1.4) * faiss_scores
        else:
            final_scores = alpha * bm25_scores + (beta * 1.1) * faiss_scores


        top_k_indices = np.argsort(final_scores)[::-1][:k]
        results = [self.vector_store.chunks[i] for i in top_k_indices]

        return results

    # ---------------------------
    # 🔥 RERANK (ADD THIS BACK)
    # ---------------------------
    def rerank(self, query, chunks, llm):
        if len(chunks) <= 1:
            return chunks

        scored = []

        for chunk in chunks:
            text = chunk["content"] if isinstance(chunk, dict) else chunk

            prompt = f"""
You are ranking how useful a text is for answering a question.

Question: {query}

Text:
{text}

Instructions:
- Return ONLY a number between 0 and 1
- 1 = highly relevant
- 0 = not relevant

Score:
"""

            try:
                score_text = llm.generate(prompt)

                match = re.search(r"\d*\.?\d+", score_text)
                score = float(match.group()) if match else 0.6

            except:
                score = 0.5

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [item[1] for item in scored]

    # ---------------------------
    # OPTIONAL (if used)
    # ---------------------------
    def get_dynamic_weights(self, query):
        query = query.lower()

        if len(query.split()) <= 3:
            return 0.7, 0.3

        elif "explain" in query or "why" in query:
            return 0.3, 0.7

        elif "compare" in query or "difference" in query:
            return 0.5, 0.5

        return 0.5, 0.5


# # hybrid_retriever.py
# import numpy as np

# class HybridRetriever:
#     def __init__(self, bm25, vector_store, embedder):
#         self.bm25 = bm25
#         self.vector_store = vector_store
#         self.embedder = embedder

#     def search(self, query, k=3, alpha=0.5, beta=0.5):
#         # ---------------------------
#         # BM25
#         # ---------------------------
#         tokenized_query = query.split()
#         bm25_scores = self.bm25.bm25.get_scores(tokenized_query)
#         bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)

#         # ---------------------------
#         # FAISS
#         # ---------------------------
#         query_vector = self.embedder.encode([query])[0]
#         faiss_results = self.vector_store.search(query_vector, is_vector=True, k=len(self.vector_store.chunks))

#         faiss_scores = np.zeros(len(self.vector_store.chunks))
#         for i, chunk in enumerate(self.vector_store.chunks):
#             chunk_vector = self.vector_store.get_vector(chunk)
#             faiss_scores[i] = float(np.dot(query_vector, chunk_vector.T))

#         # Normalize FAISS scores
#         faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores) + 1e-8)

#         # ---------------------------
#         # Combine BM25 + FAISS
#         # ---------------------------
#         final_scores = alpha * bm25_scores + beta * faiss_scores

#         top_k_indices = np.argsort(final_scores)[::-1][:k]
#         results = [self.vector_store.chunks[i] for i in top_k_indices]
#         return results

#     def rerank(self, query, chunks, llm):
#         if len(chunks) <= 1:
#             return chunks

#         scored = []
#         for chunk in chunks:
#             text = chunk["content"] if isinstance(chunk, dict) else chunk

#             prompt = f"""
# You are ranking how useful a text is for answering a question.

# Question: {query}

# Text:
# {text}

# Instructions:
# - Return ONLY a number between 0 and 1
# - 1 = highly relevant
# - 0 = not relevant

# Score:
# """
#             try:
#                 score_text = llm.generate(prompt)
#                 import re
#                 match = re.search(r"\d*\.?\d+", score_text)
#                 score = float(match.group()) if match else 0.5
#             except:
#                 score = 0.5

#             scored.append((score, chunk))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         return [item[1] for item in scored]

#     def get_dynamic_weights(self, query):
#         query = query.lower()
#         # keyword-heavy → BM25
#         if len(query.split()) <= 3:
#             return 0.7, 0.3
#         # explanation → semantic
#         elif "explain" in query or "why" in query:
#             return 0.3, 0.7
#         # comparison → balanced
#         elif "compare" in query or "difference" in query:
#             return 0.5, 0.5
#         # default
#         return 0.5, 0.5

##BESTTT
# import numpy as np

# class HybridRetriever:
#     def __init__(self, bm25, vector_store, embedder):
#         self.bm25 = bm25
#         self.vector_store = vector_store
#         self.embedder = embedder

#     def search(self, query, k=3, alpha=0.5, beta=0.5):
#         # ---------------------------
#         # BM25
#         # ---------------------------
#         tokenized_query = query.split()
#         bm25_scores = self.bm25.bm25.get_scores(tokenized_query)

#         # ---------------------------
#         # FAISS
#         # ---------------------------
#         # query_vector = self.embedder.encode([query])[0]
#         # distances, indices = self.vector_store.index.search(
#         #     np.array([query_vector]).astype("float32"),
#         #     len(self.vector_store.chunks)
#         # )
        
#         # ---------------------------
#         # FAISS
#         # ---------------------------
#         import faiss

#         query_vector = self.embedder.encode([query])[0]

#         query_vector = np.array([query_vector]).astype("float32")
#         faiss.normalize_L2(query_vector)

#         faiss_k = max(k * 3, 10)

#         distances, indices = self.vector_store.index.search(
#             query_vector,
#             faiss_k
#         )

#         # Initialize full FAISS score array
#         faiss_scores_full = np.zeros(len(self.vector_store.chunks))

#         # Fill only retrieved indices
#         for idx, score in zip(indices[0], distances[0]):
#             faiss_scores_full[idx] = score

#         faiss_scores = faiss_scores_full


#         # ---------------------------
#         # NORMALIZE
#         # ---------------------------
#         bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
#         faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores) + 1e-8)

#         # ---------------------------
#         # COMBINE
#         # ---------------------------
#         final_scores = alpha * bm25_scores + beta * faiss_scores

#         # ---------------------------
#         # TOP-K
#         # ---------------------------
#         top_k_indices = np.argsort(final_scores)[::-1][:k]
#         results = [self.vector_store.chunks[i] for i in top_k_indices]

#         return results

#     def rerank(self, query, chunks, llm):
#         if len(chunks) <= 1:
#             return chunks

#         scored = []

#         for chunk in chunks:
#             text = chunk["content"] if isinstance(chunk, dict) else chunk

#             prompt = f"""
#     You are ranking how useful a text is for answering a question.

#     Question: {query}

#     Text:
#     {text}

#     Instructions:
#     - Return ONLY a number between 0 and 1
#     - 1 = highly relevant
#     - 0 = not relevant

#     Score:
#     """

#             try:
#                 score_text = llm.generate(prompt)

#                 # 🔥 robust parsing
#                 import re
#                 match = re.search(r"\d*\.?\d+", score_text)
#                 score = float(match.group()) if match else 0.5

#             except:
#                 score = 0.5

#             scored.append((score, chunk))

#         scored.sort(key=lambda x: x[0], reverse=True)

#         return [item[1] for item in scored]

#     def get_dynamic_weights(self, query):
#         query = query.lower()

#         # keyword-heavy → BM25
#         if len(query.split()) <= 3:
#             return 0.7, 0.3

#         # explanation → semantic
#         elif "explain" in query or "why" in query:
#             return 0.3, 0.7

#         # comparison → balanced
#         elif "compare" in query or "difference" in query:
#             return 0.5, 0.5

#         # default
#         return 0.5, 0.5






# modules/rag/hybrid_retriever.py
   # 🔥 NEW: RERANK FUNCTION
#     def rerank(self, query, chunks, llm):
#         if len(chunks) <= 1:
#             return chunks

#         scored = []

#         for chunk in chunks:
#             text = chunk["content"] if isinstance(chunk, dict) else chunk

#             prompt = f"""
# Rate how relevant this text is for answering the question.

# Question: {query}

# Text:
# {text}

# Give a score from 1 (least relevant) to 10 (most relevant).
# Only output the number.
# """
#             try:
#                 score_text = llm.generate(prompt)
#                 score = float(score_text.strip()[0])  # simple parsing
#             except:
#                 score = 5  # fallback

#             scored.append((score, chunk))

#         # sort descending
#         scored.sort(key=lambda x: x[0], reverse=True)

#         return [item[1] for item in scored]
    
    # import numpy as np

# class HybridRetriever:
#     def __init__(self, bm25, vector_store, embedder):
#         self.bm25 = bm25
#         self.vector_store = vector_store
#         self.embedder = embedder

#     def search(self, query, k=2, alpha=0.5, beta=0.5):
#         # Step 1: BM25 scores
#         tokenized_query = query.lower().split()
#         bm25_scores = self.bm25.bm25.get_scores(tokenized_query)

#         # Step 2: FAISS scores
#         query_vector = self.embedder.encode([query])[0]
#         distances, indices = self.vector_store.index.search(
#             np.array([query_vector]).astype("float32"), len(self.vector_store.chunks)
#         )

#         # Convert distance to similarity
#         # faiss_scores = -distances[0]
#         faiss_scores = 1 / (1 + distances[0])

#         # Step 3: Normalize scores
#         bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
#         faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores) + 1e-8)

#         # Step 4: Combine scores
#         final_scores = alpha * bm25_scores + beta * faiss_scores

#         # Step 5: Get top-k
#         top_k_indices = np.argsort(final_scores)[::-1][:k]

#         results = [self.vector_store.chunks[i] for i in top_k_indices]

#         return results