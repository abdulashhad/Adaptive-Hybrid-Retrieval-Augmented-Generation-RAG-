# from rank_bm25 import BM25Okapi

# class BM25Retriever:
#     def __init__(self, chunks):
#         # Extract text
#         self.texts = [chunk["content"] for chunk in chunks]

#         # Tokenize (simple split)
#         self.tokenized_corpus = [text.split() for text in self.texts]

#         # Create BM25 model
#         self.bm25 = BM25Okapi(self.tokenized_corpus)

#         self.chunks = chunks

#         print("[INFO] BM25 Retriever initialized")

#     def search(self, query, k=2):
#         tokenized_query = query.split()

#         scores = self.bm25.get_scores(tokenized_query)

#         # Get top-k indices
#         top_k_indices = sorted(
#             range(len(scores)),
#             key=lambda i: scores[i],
#             reverse=True
#         )[:k]

#         results = [self.chunks[i] for i in top_k_indices]

#         return results


# modules/rag/bm25_retriever.py

from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        self.texts = [chunk["content"] for chunk in chunks]

        self.tokenized_corpus = [self.preprocess(text) for text in self.texts]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("[INFO] BM25 Retriever initialized")

    # ---------------------------
    # TEXT PREPROCESSING GOOOODDDDD
    # ---------------------------
    # def preprocess(self, text):
    #     text = text.lower()
    #     text = re.sub(r"[^\w\s]", "", text)
    #     tokens = text.split()

    #     stopwords = {
    #         "the", "is", "in", "and", "to", "of", "a",
    #         "for", "on", "with", "as", "by", "an"
    #     }

    #     tokens = [t for t in tokens if t not in stopwords]

    #     return tokens

    def preprocess(self, text):
        text = text.lower()

        # keep important ML words intact
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        tokens = text.split()

        # LIGHTER stopword removal (important!)
        stopwords = {"the", "is", "in", "and", "to", "of"}

        tokens = [t for t in tokens if t not in stopwords]

        return tokens


    # ---------------------------
    # SEARCH
    # ---------------------------
    def search(self, query, k=2):
        tokenized_query = self.preprocess(query)  # ✅ FIX

        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = [self.chunks[i] for i in top_k_indices]

        return results


##besttt
# from rank_bm25 import BM25Okapi
# import re

# class BM25Retriever:
#     def __init__(self, chunks):
#         self.chunks = chunks

#         # Extract text
#         self.texts = [chunk["content"] for chunk in chunks]

#         # Preprocess corpus
#         self.tokenized_corpus = [self.preprocess(text) for text in self.texts]

#         # Create BM25 model
#         self.bm25 = BM25Okapi(self.tokenized_corpus)

#         print("[INFO] BM25 Retriever initialized")

#     # ---------------------------
#     # TEXT PREPROCESSING 🔥
#     # ---------------------------
#     def preprocess(self, text):
#         text = text.lower()  # normalize case
#         text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
#         tokens = text.split()

#         # optional: remove stopwords (light version)
#         stopwords = {
#             "the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "as", "by", "an"
#         }

#         tokens = [t for t in tokens if t not in stopwords]

#         return tokens

#     # ---------------------------
#     # SEARCH
#     # ---------------------------
#     def search(self, query, k=2):
#         tokenized_query = self.preprocess(query)

#         scores = self.bm25.get_scores(tokenized_query)

#         # Get top-k indices
#         top_k_indices = sorted(
#             range(len(scores)),
#             key=lambda i: scores[i],
#             reverse=True
#         )[:k]

#         results = [self.chunks[i] for i in top_k_indices]

#         return results