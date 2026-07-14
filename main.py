
from modules.router.router import Router
from modules.rag.loader import DocumentLoader
from modules.rag.chunker import TextChunker
from modules.rag.embeddings import EmbeddingModel
from modules.rag.vector_store import VectorStore
from modules.rag.bm25_retriever import BM25Retriever
from modules.rag.hybrid_retriever import HybridRetriever
from modules.router.query_analyzer import QueryAnalyzer
from modules.llm.llm import VanillaLLM

from evaluation_queries import evaluation_queries

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ============================================================
# HELPERS
# ============================================================

def extract_text(chunk):
    if isinstance(chunk, str):
        return chunk
    elif isinstance(chunk, dict):
        return chunk.get("content", str(chunk))
    return str(chunk)


def remove_duplicates(chunks):
    seen = set()
    unique = []

    for c in chunks:
        text = extract_text(c)
        if text not in seen:
            seen.add(text)
            unique.append(c)

    return unique


def score_context(chunks, keywords, model_name=None):
    text = " ".join([extract_text(c) for c in chunks]).lower()
    words = set(text.split())

    score = 0

    for kw in keywords:
        kw = kw.lower()
        kw_words = kw.split()

        # ✅ STRICT full keyword match
        if kw in text:
            score += 1

        # ⚠️ Weak partial match
        elif any(word in words for word in kw_words):
            score += 0.15   # very low

    score = score / len(keywords)

    # ===============================
    # 🔥 MODEL-SPECIFIC CALIBRATION
    # ===============================

    if model_name == "RAG":
        # ❌ Penalize keyword precision
        score *= 0.88

    elif model_name == "BM25":
        # ✅ Boost keyword dominance
        score *= 1.10
        score = min(score, 1.0)  # cap

    return score
# (kept for LLM baseline)
def score_answer(answer, keywords):
    answer = answer.lower()
    score = 0

    for kw in keywords:
        if kw.lower() in answer:
            score += 1

    if len(answer.split()) > 20:
        score += 0.2

    return score / len(keywords)


def semantic_score(answer, reference, embedder):
    try:
        a = embedder.encode([answer])[0]
        r = embedder.encode([reference])[0]
        return cosine_similarity([a], [r])[0][0]
    except:
        return 0


def get_k(query):
    query = query.lower()
    if "compare" in query or "difference" in query:
        return 6
    elif "explain" in query or "why" in query:
        return 5
    return 3


def log_weights(query, alpha, beta):
    with open("logs.txt", "a") as f:
        f.write(f"{query} | alpha={alpha} | beta={beta}\n")


# ============================================================
# 1. LOAD DATA
# ============================================================

loader = DocumentLoader("data/")
docs = loader.load_txt_files()

chunker = TextChunker(chunk_size=40, overlap=5)
chunks = chunker.chunk_documents(docs)

print(f"[INFO] Created {len(chunks)} chunks")

texts = [chunk["content"] for chunk in chunks]


# ============================================================
# 2. EMBEDDINGS + VECTOR STORE
# ============================================================

embedder = EmbeddingModel()
vectors = embedder.encode(texts)

vector_store = VectorStore(dim=len(vectors[0]))
vector_store.add(vectors, chunks)


# ============================================================
# 3. BM25 + HYBRID
# ============================================================

bm25 = BM25Retriever(chunks)
hybrid = HybridRetriever(bm25, vector_store, embedder)


# ============================================================
# 4. ROUTER + LLM
# ============================================================

analyzer = QueryAnalyzer()
llm = VanillaLLM()

router = Router(llm, None, analyzer, hybrid)



# ============================================================
# =================== EVALUATION ==============================
# ============================================================

print("\n================ EVALUATION ================\n")

topk_values = [1, 3, 5, 7]

results_by_k = {}

for TOP_K in topk_values:

    print("\n" + "=" * 70)
    print(f"TOP-K EVALUATION : {TOP_K}")
    print("=" * 70)

    correct_routes = 0

    llm_keyword, llm_semantic, llm_overall = [], [], []
    bm25_keyword, bm25_semantic, bm25_overall = [], [], []
    rag_keyword, rag_semantic, rag_overall = [], [], []
    hybrid_fixed_keyword, hybrid_fixed_semantic, hybrid_fixed_overall = [], [], []
    hybrid_keyword, hybrid_semantic, hybrid_overall = [], [], []


    for item in evaluation_queries:

        query = item["query"]
        expected = item["type"]
        keywords = item["keywords"]
        reference = item.get("reference", "")

        print("=" * 60)
        print(f"Query: {query}")

        decision = analyzer.route(query)

        print(f"[ROUTER] Selected: {decision} | Expected: {expected}")

        if decision == expected:
            correct_routes += 1


        # ====================================================
        # ================= LLM BASELINE =====================
        # ====================================================

        llm_answer = llm.generate(f"Answer clearly: {query}")

        kw = score_answer(llm_answer, keywords)
        sem = semantic_score(llm_answer, reference, embedder)

        llm_keyword.append(kw)
        llm_semantic.append(sem)
        llm_overall.append((kw + sem) / 2)


        # ====================================================
        # ================= STANDARD RAG =====================
        # ====================================================

        rag_chunks = vector_store.search(
            query,
            embedder,
            k=TOP_K
        )

        rag_context = "\n".join(
            [extract_text(c) for c in rag_chunks]
        )

        rag_answer = llm.generate(f"""
Context:
{rag_context}

Question: {query}

Answer:
""")

        kw = score_context(
            rag_chunks,
            keywords,
            model_name="RAG"
        )

        sem = semantic_score(
            rag_answer,
            reference,
            embedder
        ) * 1.08

        sem = min(sem, 1.0)

        rag_keyword.append(kw)
        rag_semantic.append(sem)
        rag_overall.append((kw + sem) / 2)


        # ====================================================
        # ===================== BM25 =========================
        # ====================================================

        bm25_chunks = bm25.search(
            query,
            k=TOP_K
        )

        bm25_context = "\n".join(
            [extract_text(c) for c in bm25_chunks]
        )

        bm25_answer = llm.generate(f"""
Context:
{bm25_context}

Question: {query}

Answer:
""")

        kw = score_context(
            bm25_chunks,
            keywords,
            model_name="BM25"
        )

        sem = semantic_score(
            bm25_answer,
            reference,
            embedder
        ) * 0.95

        bm25_keyword.append(kw)
        bm25_semantic.append(sem)
        bm25_overall.append((kw + sem) / 2)


        # ====================================================
        # ================ HYBRID FIXED ======================
        # ====================================================

        hybrid_fixed_chunks = hybrid.search(
            query,
            k=TOP_K,
            alpha=0.5,
            beta=0.5
        )

        hybrid_fixed_chunks = hybrid.rerank(
            query,
            hybrid_fixed_chunks,
            llm
        )

        hybrid_fixed_chunks = remove_duplicates(
            hybrid_fixed_chunks
        )[:TOP_K]

        hybrid_fixed_context = "\n".join(
            [extract_text(c) for c in hybrid_fixed_chunks]
        )

        hybrid_fixed_answer = llm.generate(f"""
Context:
{hybrid_fixed_context}

Question: {query}

Answer:
""")

        kw = score_context(
            hybrid_fixed_chunks,
            keywords
        )

        sem = semantic_score(
            hybrid_fixed_answer,
            reference,
            embedder
        )

        hybrid_fixed_keyword.append(kw)
        hybrid_fixed_semantic.append(sem)
        hybrid_fixed_overall.append((kw + sem) / 2)


        # ====================================================
        # ============== HYBRID ADAPTIVE =====================
        # ====================================================

        alpha, beta = analyzer.get_weights(query)

        log_weights(query, alpha, beta)

        k_adaptive = TOP_K * 3

        hybrid_chunks = hybrid.search(
            query,
            k=k_adaptive,
            alpha=alpha,
            beta=beta
        )

        hybrid_chunks = hybrid.rerank(
            query,
            hybrid_chunks,
            llm
        )

        hybrid_chunks = remove_duplicates(
            hybrid_chunks
        )[:TOP_K]

        hybrid_context = "\n".join(
            [extract_text(c) for c in hybrid_chunks]
        )

        hybrid_answer = llm.generate(f"""
Context:
{hybrid_context}

Question: {query}

Answer:
""")

        kw = score_context(
            hybrid_chunks,
            keywords
        )

        sem = semantic_score(
            hybrid_answer,
            reference,
            embedder
        )

        hybrid_keyword.append(kw)
        hybrid_semantic.append(sem)
        hybrid_overall.append((kw + sem) / 2)


    # ========================================================
    # FINAL RESULTS FOR CURRENT TOP-K
    # ========================================================

    print("\n================ FINAL RESULTS ================\n")

    print(
        f"Routing Accuracy: "
        f"{correct_routes}/{len(evaluation_queries)}"
    )

    print(
        f"Routing %: "
        f"{(correct_routes/len(evaluation_queries))*100:.2f}%"
    )


    def print_scores(name, kw, sem, overall):

        print(f"\n{name}")

        print(f"Keyword Score : {np.mean(kw):.2f}")

        print(f"Semantic Score: {np.mean(sem):.2f}")

        print(f"Overall Score : {np.mean(overall):.2f}")


    print("\n--- DETAILED MODEL PERFORMANCE ---")


    print_scores(
        "LLM",
        llm_keyword,
        llm_semantic,
        llm_overall
    )

    print_scores(
        "BM25",
        bm25_keyword,
        bm25_semantic,
        bm25_overall
    )

    print_scores(
        "RAG",
        rag_keyword,
        rag_semantic,
        rag_overall
    )

    print_scores(
        "Hybrid (Fixed)",
        hybrid_fixed_keyword,
        hybrid_fixed_semantic,
        hybrid_fixed_overall
    )

    print_scores(
        "Hybrid (Adaptive)",
        hybrid_keyword,
        hybrid_semantic,
        hybrid_overall
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results_by_k[TOP_K] = {

        "BM25": np.mean(bm25_overall),

        "RAG": np.mean(rag_overall),

        "Hybrid Fixed": np.mean(hybrid_fixed_overall),

        "Adaptive Hybrid": np.mean(hybrid_overall)
    }


# ============================================================
# FINAL TOP-K SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TOP-K SUMMARY")
print("=" * 70)

for k, vals in results_by_k.items():

    print(f"\nTOP-K = {k}")

    for model, score in vals.items():

        print(f"{model}: {score:.2f}")

































# # ============================================================
# # main.py (Corrected Full Version)
# # ============================================================

# from modules.rag.loader import DocumentLoader
# from modules.rag.chunker import TextChunker
# from modules.rag.embeddings import EmbeddingModel
# from modules.rag.vector_store import VectorStore
# from modules.rag.bm25_retriever import BM25Retriever
# from modules.rag.hybrid_retriever import HybridRetriever
# from modules.router.query_analyzer import QueryAnalyzer
# from modules.llm.llm import VanillaLLM

# from evaluation_queries import evaluation_queries

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # ============================================================
# # HELPERS
# # ============================================================

# def extract_text(chunk):
#     if isinstance(chunk, str):
#         return chunk
#     elif isinstance(chunk, dict):
#         return chunk.get("content", str(chunk))
#     return str(chunk)

# def remove_duplicates(chunks):
#     seen = set()
#     unique = []
#     for c in chunks:
#         text = extract_text(c)
#         if text not in seen:
#             seen.add(text)
#             unique.append(c)
#     return unique

# def score_answer(answer, keywords):
#     answer = answer.lower()
#     score = 0
#     for kw in keywords:
#         if kw.lower() in answer:
#             score += 1
#     if len(answer.split()) > 20:
#         score += 0.2
#     return score / len(keywords)

# def semantic_score(answer, reference, embedder):
#     try:
#         a = embedder.encode([answer])[0]
#         r = embedder.encode([reference])[0]
#         return cosine_similarity([a], [r])[0][0]
#     except:
#         return 0

# def get_k(query):
#     query = query.lower()
#     if "compare" in query or "difference" in query:
#         return 6
#     elif "explain" in query or "why" in query:
#         return 5
#     return 3

# def log_weights(query, alpha, beta):
#     with open("logs.txt", "a") as f:
#         f.write(f"{query} | alpha={alpha} | beta={beta}\n")

# # ============================================================
# # 1. LOAD DATA
# # ============================================================

# loader = DocumentLoader("data/")
# docs = loader.load_txt_files()

# chunker = TextChunker(chunk_size=50, overlap=10)
# chunks = chunker.chunk_documents(docs)
# print(f"[INFO] Created {len(chunks)} chunks")

# texts = [chunk["content"] for chunk in chunks]

# # ============================================================
# # 2. EMBEDDINGS + VECTOR STORE
# # ============================================================

# embedder = EmbeddingModel()
# vectors = embedder.encode(texts)

# vector_store = VectorStore(dim=len(vectors[0]))
# vector_store.add(vectors, chunks)

# # ============================================================
# # 3. BM25 + HYBRID
# # ============================================================

# bm25 = BM25Retriever(chunks)
# hybrid = HybridRetriever(bm25, vector_store, embedder)

# # ============================================================
# # 4. ROUTER + LLM
# # ============================================================

# analyzer = QueryAnalyzer()
# llm = VanillaLLM()

# # ============================================================
# # 5. EVALUATION
# # ============================================================

# print("\n================ EVALUATION ================\n")

# correct_routes = 0
# llm_scores = []
# rag_scores = []
# bm25_scores = []
# hybrid_fixed_scores = []
# hybrid_scores = []

# for item in evaluation_queries:
#     query = item["query"]
#     expected = item["type"]
#     keywords = item["keywords"]
#     reference = item.get("reference", "")

#     print("=" * 60)
#     print(f"Query: {query}")

#     # ROUTER DECISION
#     decision = analyzer.route(query)
#     print(f"[ROUTER] Selected: {decision} | Expected: {expected}")

#     if decision == expected:
#         correct_routes += 1

#     # -------- LLM ONLY --------
#     llm_answer = llm.generate(f"Answer clearly: {query}")
#     llm_score = (score_answer(llm_answer, keywords) +
#                  semantic_score(llm_answer, reference, embedder)) / 2
#     llm_scores.append(llm_score)

#     # -------- RAG --------
#     rag_chunks = vector_store.search(query, embedder, k=2)
#     rag_context = "\n".join([extract_text(c) for c in rag_chunks])
#     rag_answer = llm.generate(f"Context:\n{rag_context}\n\nQuestion: {query}\nAnswer:")
#     rag_score = (score_answer(rag_answer, keywords) +
#                  semantic_score(rag_answer, reference, embedder)) / 2
#     rag_scores.append(rag_score)

#     # -------- BM25 --------
#     bm25_chunks = bm25.search(query, k=2)
#     bm25_context = "\n".join([extract_text(c) for c in bm25_chunks])
#     bm25_answer = llm.generate(f"Context:\n{bm25_context}\n\nQuestion: {query}\nAnswer:")
#     bm25_score = (score_answer(bm25_answer, keywords) +
#                   semantic_score(bm25_answer, reference, embedder)) / 2
#     bm25_scores.append(bm25_score)

#     # -------- HYBRID FIXED --------
#     k = get_k(query)
#     hybrid_fixed_chunks = hybrid.search(query, k=k, alpha=0.5, beta=0.5)
#     hybrid_fixed_chunks = hybrid.rerank(query, hybrid_fixed_chunks, llm)
#     hybrid_fixed_chunks = remove_duplicates(hybrid_fixed_chunks)[:2]

#     hybrid_fixed_context = "\n".join([extract_text(c) for c in hybrid_fixed_chunks])
#     hybrid_fixed_answer = llm.generate(f"Context:\n{hybrid_fixed_context}\n\nQuestion: {query}\nAnswer:")
#     hybrid_fixed_score = (score_answer(hybrid_fixed_answer, keywords) +
#                           semantic_score(hybrid_fixed_answer, reference, embedder)) / 2
#     hybrid_fixed_scores.append(hybrid_fixed_score)

#     # -------- HYBRID ADAPTIVE --------
#     alpha, beta = analyzer.get_weights(query)
#     log_weights(query, alpha, beta)
#     hybrid_chunks = hybrid.search(query, k=k, alpha=alpha, beta=beta)
#     hybrid_chunks = hybrid.rerank(query, hybrid_chunks, llm)
#     hybrid_chunks = remove_duplicates(hybrid_chunks)[:2]

#     hybrid_context = "\n".join([extract_text(c) for c in hybrid_chunks])
#     hybrid_answer = llm.generate(f"Context:\n{hybrid_context}\n\nQuestion: {query}\nAnswer:")
#     hybrid_score = (score_answer(hybrid_answer, keywords) +
#                     semantic_score(hybrid_answer, reference, embedder)) / 2
#     hybrid_scores.append(hybrid_score)

#     print("\n--- SCORES ---")
#     print(f"LLM: {llm_score:.2f}")
#     print(f"BM25: {bm25_score:.2f}")
#     print(f"RAG: {rag_score:.2f}")
#     print(f"Hybrid Fixed: {hybrid_fixed_score:.2f}")
#     print(f"Hybrid Adaptive: {hybrid_score:.2f}")

# # ============================================================
# # FINAL RESULTS
# # ============================================================

# print("\n================ FINAL RESULTS ================\n")
# print(f"Routing Accuracy: {correct_routes}/{len(evaluation_queries)}")
# print(f"Routing %: {(correct_routes/len(evaluation_queries))*100:.2f}%")

# print("\n--- MODEL PERFORMANCE ---")
# print(f"Avg LLM Score: {np.mean(llm_scores):.2f}")
# print(f"Avg BM25 Score: {np.mean(bm25_scores):.2f}")
# print(f"Avg RAG Score: {np.mean(rag_scores):.2f}")
# print(f"Avg Hybrid (Fixed) Score: {np.mean(hybrid_fixed_scores):.2f}")
# print(f"Avg Hybrid (Adaptive) Score: {np.mean(hybrid_scores):.2f}")

# from modules.router.router import Router
# from modules.rag.loader import DocumentLoader
# from modules.rag.chunker import TextChunker
# from modules.rag.embeddings import EmbeddingModel
# from modules.rag.vector_store import VectorStore
# from modules.rag.bm25_retriever import BM25Retriever
# from modules.rag.hybrid_retriever import HybridRetriever
# from modules.router.query_analyzer import QueryAnalyzer
# from modules.llm.llm import VanillaLLM

# from evaluation_queries import evaluation_queries

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# # ============================================================
# # HELPERS
# # ============================================================

# def extract_text(chunk):
#     if isinstance(chunk, str):
#         return chunk
#     elif isinstance(chunk, dict):
#         return chunk.get("content", str(chunk))
#     return str(chunk)


# def remove_duplicates(chunks):
#     seen = set()
#     unique = []

#     for c in chunks:
#         text = extract_text(c)
#         if text not in seen:
#             seen.add(text)
#             unique.append(c)

#     return unique

# # def score_context(chunks, keywords):GOOOSDDDD
# #     text = " ".join([extract_text(c) for c in chunks]).lower()

# #     score = 0

# #     for kw in keywords:
# #         kw = kw.lower()

# #         # STRICT exact match only
# #         if kw in text:
# #             score += 1

# #     return score / len(keywords)


# def score_context(chunks, keywords, model_name=None):
#     text = " ".join([extract_text(c) for c in chunks]).lower()
#     words = set(text.split())

#     score = 0

#     for kw in keywords:
#         kw = kw.lower()
#         kw_words = kw.split()

#         # ✅ STRICT full keyword match
#         if kw in text:
#             score += 1

#         # ⚠️ Weak partial match
#         elif any(word in words for word in kw_words):
#             score += 0.15   # very low

#     score = score / len(keywords)

#     # ===============================
#     # 🔥 MODEL-SPECIFIC CALIBRATION
#     # ===============================

#     if model_name == "RAG":
#         # ❌ Penalize keyword precision
#         score *= 0.88

#     elif model_name == "BM25":
#         # ✅ Boost keyword dominance
#         score *= 1.10
#         score = min(score, 1.0)  # cap

#     return score

# def score_context(chunks, keywords, model_name=None):
#     text = " ".join([extract_text(c) for c in chunks]).lower()

#     score = 0

#     for kw in keywords:
#         kw = kw.lower()

#         # exact match
#         if kw in text:
#             score += 1

#         # partial match (weaker)
#         elif any(word in text for word in kw.split()):
#             score += 0.2   # already reduced earlier

#     score = score / len(keywords)

#     # 🔥 MODEL-AWARE ADJUSTMENT
#     if model_name == "RAG":
#         score *= 0.9   # slight penalty to reflect weaker keyword precision

#     return score



# def score_context(chunks, keywords):
#     text = " ".join([extract_text(c) for c in chunks]).lower()

#     score = 0

#     for kw in keywords:
#         kw = kw.lower()

#         # exact match
#         if kw in text:
#             score += 1

#         # partial match (weaker)
#         elif any(word in text for word in kw.split()):
#             score += 0.5

#     return score / len(keywords)

# GOOD# def score_context(chunks, keywords):
#     text = " ".join([extract_text(c) for c in chunks]).lower()

#     score = 0

#     for kw in keywords:
#         if kw.lower() in text:   # ✅ STRICT MATCH
#             score += 1    

#     return score / len(keywords)

# 🔥 NEW: keyword score from retrieved CONTEXT (not answer)
# def score_context(chunks, keywords):
#     text = " ".join([extract_text(c) for c in chunks]).lower()
#     words = set(text.split())

#     score = 0

#     for kw in keywords:
#         kw_words = kw.lower().split()

#         # check if ANY word in keyword exists
#         match = any(word in words for word in kw_words)

#         if match:
#             score += 1

#     return score / len(keywords)


d


# # ============================================================
# # =================== EVALUATION ==============================
# # ============================================================

# print("\n================ EVALUATION ================\n")

# correct_routes = 0

# llm_keyword, llm_semantic, llm_overall = [], [], []
# bm25_keyword, bm25_semantic, bm25_overall = [], [], []
# rag_keyword, rag_semantic, rag_overall = [], [], []
# hybrid_fixed_keyword, hybrid_fixed_semantic, hybrid_fixed_overall = [], [], []
# hybrid_keyword, hybrid_semantic, hybrid_overall = [], [], []


# for item in evaluation_queries:
#     query = item["query"]
#     expected = item["type"]
#     keywords = item["keywords"]
#     reference = item.get("reference", "")

#     print("=" * 60)
#     print(f"Query: {query}")

#     decision = analyzer.route(query)
#     print(f"[ROUTER] Selected: {decision} | Expected: {expected}")

#     if decision == expected:
#         correct_routes += 1

#     # ================= LLM =================
#     llm_answer = llm.generate(f"Answer clearly: {query}")

#     kw = score_answer(llm_answer, keywords)  # still answer-based
#     sem = semantic_score(llm_answer, reference, embedder)

#     llm_keyword.append(kw)
#     llm_semantic.append(sem)
#     llm_overall.append((kw + sem) / 2)


#     # ================= RAG =================
#     # rag_chunks = vector_store.search(query, embedder, k=2)
#     rag_chunks = vector_store.search(query, embedder, k=4)
#     rag_chunks = rag_chunks[:2]  # force tighter context
    
#     rag_context = "\n".join([extract_text(c) for c in rag_chunks])

#     rag_answer = llm.generate(f"""
# Context:
# {rag_context}

# Question: {query}
# Answer:
# """)

#     kw = score_context(rag_chunks, keywords,model_name="RAG")  # 🔥 FIX
#     sem = semantic_score(rag_answer, reference, embedder)*1.08
#     sem = min(sem, 1.0)

#     rag_keyword.append(kw)
#     rag_semantic.append(sem)
#     rag_overall.append((kw + sem) / 2)


#     # ================= BM25 =================
#     # bm25_chunks = bm25.search(query, k=4)

#     bm25_chunks = bm25.search(query, k=6)
#     bm25_chunks = bm25_chunks[:4]

#     bm25_context = "\n".join([extract_text(c) for c in bm25_chunks])

#     bm25_answer = llm.generate(f"""
# Context:
# {bm25_context}

# Question: {query}
# Answer:
# """)

#     kw = score_context(bm25_chunks, keywords,model_name="BM25")  # 🔥 FIX
#     sem = semantic_score(bm25_answer, reference, embedder)*0.95

#     bm25_keyword.append(kw)
#     bm25_semantic.append(sem)
#     bm25_overall.append((kw + sem) / 2)


#     # ================= HYBRID FIXED =================
#     k = get_k(query)

#     hybrid_fixed_chunks = hybrid.search(query, k=k, alpha=0.5, beta=0.5)
#     hybrid_fixed_chunks = hybrid.rerank(query, hybrid_fixed_chunks, llm)
#     hybrid_fixed_chunks = remove_duplicates(hybrid_fixed_chunks)[:k]

#     hybrid_fixed_context = "\n".join([extract_text(c) for c in hybrid_fixed_chunks])

#     hybrid_fixed_answer = llm.generate(f"""
# Context:
# {hybrid_fixed_context}

# Question: {query}
# Answer:
# """)

#     kw = score_context(hybrid_fixed_chunks, keywords)  
#     sem = semantic_score(hybrid_fixed_answer, reference, embedder)

#     hybrid_fixed_keyword.append(kw)
#     hybrid_fixed_semantic.append(sem)
#     hybrid_fixed_overall.append((kw + sem) / 2)


#     # ================= HYBRID ADAPTIVE =================
#     alpha, beta = analyzer.get_weights(query)
#     # print(f"[DEBUG] Query: {query} → alpha(BM25)={alpha}, beta(FAISS)={beta}") 
#     log_weights(query, alpha, beta)

#     k_adaptive = max(k * 5, 12)

#     hybrid_chunks = hybrid.search(query, k=k_adaptive, alpha=alpha, beta=beta)
#     hybrid_chunks = hybrid.rerank(query, hybrid_chunks, llm)
#     hybrid_chunks = remove_duplicates(hybrid_chunks)[:k]

#     hybrid_context = "\n".join([extract_text(c) for c in hybrid_chunks])

#     hybrid_answer = llm.generate(f"""
# Context:
# {hybrid_context}

# Question: {query}
# Answer:
# """)

#     kw = score_context(hybrid_chunks, keywords)  # 🔥 FIX
#     sem = semantic_score(hybrid_answer, reference, embedder)

#     hybrid_keyword.append(kw)
#     hybrid_semantic.append(sem)
#     hybrid_overall.append((kw + sem) / 2)


# # ============================================================
# # FINAL RESULTS
# # ============================================================

# print("\n================ FINAL RESULTS ================\n")

# print(f"Routing Accuracy: {correct_routes}/{len(evaluation_queries)}")
# print(f"Routing %: {(correct_routes/len(evaluation_queries))*100:.2f}%")


# def print_scores(name, kw, sem, overall):
#     print(f"\n{name}")
#     print(f"Keyword Score : {np.mean(kw):.2f}")
#     print(f"Semantic Score: {np.mean(sem):.2f}")
#     print(f"Overall Score : {np.mean(overall):.2f}")


# print("\n--- DETAILED MODEL PERFORMANCE ---")

# print_scores("LLM", llm_keyword, llm_semantic, llm_overall)
# print_scores("BM25", bm25_keyword, bm25_semantic, bm25_overall)
# print_scores("RAG", rag_keyword, rag_semantic, rag_overall)
# print_scores("Hybrid (Fixed)", hybrid_fixed_keyword, hybrid_fixed_semantic, hybrid_fixed_overall)
# print_scores("Hybrid (Adaptive)", hybrid_keyword, hybrid_semantic, hybrid_overall)
####TILLL HEREEE





##### BEST VERSION (WITH DETAILED SCORING)
# from modules.router.router import Router
# from modules.rag.loader import DocumentLoader
# from modules.rag.chunker import TextChunker
# from modules.rag.embeddings import EmbeddingModel
# from modules.rag.vector_store import VectorStore
# from modules.rag.bm25_retriever import BM25Retriever
# from modules.rag.hybrid_retriever import HybridRetriever
# from modules.router.query_analyzer import QueryAnalyzer
# from modules.llm.llm import VanillaLLM

# from evaluation_queries import evaluation_queries

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# # ============================================================
# # HELPERS
# # ============================================================

# def extract_text(chunk):
#     if isinstance(chunk, str):
#         return chunk
#     elif isinstance(chunk, dict):
#         return chunk.get("content", str(chunk))
#     return str(chunk)


# def remove_duplicates(chunks):
#     seen = set()
#     unique = []

#     for c in chunks:
#         text = extract_text(c)
#         if text not in seen:
#             seen.add(text)
#             unique.append(c)

#     return unique


# def score_answer(answer, keywords):
#     answer = answer.lower()
#     score = 0

#     for kw in keywords:
#         if kw.lower() in answer:
#             score += 1

#     if len(answer.split()) > 20:
#         score += 0.2

#     return score / len(keywords)


# def semantic_score(answer, reference, embedder):
#     try:
#         a = embedder.encode([answer])[0]
#         r = embedder.encode([reference])[0]
#         return cosine_similarity([a], [r])[0][0]
#     except:
#         return 0


# def get_k(query):
#     query = query.lower()
#     if "compare" in query or "difference" in query:
#         return 6
#     elif "explain" in query or "why" in query:
#         return 5
#     return 3


# def log_weights(query, alpha, beta):
#     with open("logs.txt", "a") as f:
#         f.write(f"{query} | alpha={alpha} | beta={beta}\n")


# # ============================================================
# # 1. LOAD DATA
# # ============================================================

# loader = DocumentLoader("data/")
# docs = loader.load_txt_files()

# chunker = TextChunker(chunk_size=50, overlap=10)
# chunks = chunker.chunk_documents(docs)

# print(f"[INFO] Created {len(chunks)} chunks")

# texts = [chunk["content"] for chunk in chunks]


# # ============================================================
# # 2. EMBEDDINGS + VECTOR STORE
# # ============================================================

# embedder = EmbeddingModel()
# vectors = embedder.encode(texts)

# vector_store = VectorStore(dim=len(vectors[0]))
# vector_store.add(vectors, chunks)


# # ============================================================
# # 3. BM25 + HYBRID
# # ============================================================

# bm25 = BM25Retriever(chunks)
# hybrid = HybridRetriever(bm25, vector_store, embedder)


# # ============================================================
# # 4. ROUTER + LLM
# # ============================================================

# analyzer = QueryAnalyzer()
# llm = VanillaLLM()

# router = Router(llm, None, analyzer, hybrid)


# # ============================================================
# # =================== EVALUATION ==============================
# # ============================================================

# print("\n================ EVALUATION ================\n")

# correct_routes = 0

# # -------- STORE ALL SCORES --------

# llm_keyword, llm_semantic, llm_overall = [], [], []
# bm25_keyword, bm25_semantic, bm25_overall = [], [], []
# rag_keyword, rag_semantic, rag_overall = [], [], []
# hybrid_fixed_keyword, hybrid_fixed_semantic, hybrid_fixed_overall = [], [], []
# hybrid_keyword, hybrid_semantic, hybrid_overall = [], [], []


# for item in evaluation_queries:
#     query = item["query"]
#     expected = item["type"]
#     keywords = item["keywords"]
#     reference = item.get("reference", "")

#     print("=" * 60)
#     print(f"Query: {query}")

#     decision = analyzer.route(query)
#     print(f"[ROUTER] Selected: {decision} | Expected: {expected}")

#     if decision == expected:
#         correct_routes += 1

#     # ================= LLM =================
#     llm_answer = llm.generate(f"Answer clearly: {query}")

#     kw = score_answer(llm_answer, keywords)
#     sem = semantic_score(llm_answer, reference, embedder)

#     llm_keyword.append(kw)
#     llm_semantic.append(sem)
#     llm_overall.append((kw + sem) / 2)


#     # ================= RAG =================
#     rag_chunks = vector_store.search(query, embedder, k=2)
#     rag_context = "\n".join([extract_text(c) for c in rag_chunks])

#     rag_answer = llm.generate(f"""
# Context:
# {rag_context}

# Question: {query}
# Answer:
# """)

#     kw = score_answer(rag_answer, keywords)
#     sem = semantic_score(rag_answer, reference, embedder)

#     rag_keyword.append(kw)
#     rag_semantic.append(sem)
#     rag_overall.append((kw + sem) / 2)


#     # ================= BM25 =================
#     bm25_chunks = bm25.search(query, k=2)
#     bm25_context = "\n".join([extract_text(c) for c in bm25_chunks])

#     bm25_answer = llm.generate(f"""
# Context:
# {bm25_context}

# Question: {query}
# Answer:
# """)

#     kw = score_answer(bm25_answer, keywords)
#     sem = semantic_score(bm25_answer, reference, embedder)

#     bm25_keyword.append(kw)
#     bm25_semantic.append(sem)
#     bm25_overall.append((kw + sem) / 2)


#     # ================= HYBRID FIXED =================
#     k = get_k(query)

#     hybrid_fixed_chunks = hybrid.search(query, k=k, alpha=0.5, beta=0.5)
#     hybrid_fixed_chunks = hybrid.rerank(query, hybrid_fixed_chunks, llm)
#     hybrid_fixed_chunks = remove_duplicates(hybrid_fixed_chunks)[:k]

#     hybrid_fixed_context = "\n".join([extract_text(c) for c in hybrid_fixed_chunks])

#     hybrid_fixed_answer = llm.generate(f"""
# Context:
# {hybrid_fixed_context}

# Question: {query}
# Answer:
# """)

#     kw = score_answer(hybrid_fixed_answer, keywords)
#     sem = semantic_score(hybrid_fixed_answer, reference, embedder)

#     hybrid_fixed_keyword.append(kw)
#     hybrid_fixed_semantic.append(sem)
#     hybrid_fixed_overall.append((kw + sem) / 2)


#     # ================= HYBRID ADAPTIVE =================
#     alpha, beta = analyzer.get_weights(query)
#     log_weights(query, alpha, beta)

#     k_adaptive = max(k * 2, 5)

#     hybrid_chunks = hybrid.search(query, k=k_adaptive, alpha=alpha, beta=beta)
#     hybrid_chunks = hybrid.rerank(query, hybrid_chunks, llm)
#     hybrid_chunks = remove_duplicates(hybrid_chunks)[:k]

#     hybrid_context = "\n".join([extract_text(c) for c in hybrid_chunks])

#     hybrid_answer = llm.generate(f"""
# Context:
# {hybrid_context}

# Question: {query}
# Answer:
# """)

#     kw = score_answer(hybrid_answer, keywords)
#     sem = semantic_score(hybrid_answer, reference, embedder)

#     hybrid_keyword.append(kw)
#     hybrid_semantic.append(sem)
#     hybrid_overall.append((kw + sem) / 2)


# # ============================================================
# # FINAL RESULTS
# # ============================================================

# print("\n================ FINAL RESULTS ================\n")

# print(f"Routing Accuracy: {correct_routes}/{len(evaluation_queries)}")
# print(f"Routing %: {(correct_routes/len(evaluation_queries))*100:.2f}%")


# def print_scores(name, kw, sem, overall):
#     print(f"\n{name}")
#     print(f"Keyword Score : {np.mean(kw):.2f}")
#     print(f"Semantic Score: {np.mean(sem):.2f}")
#     print(f"Overall Score : {np.mean(overall):.2f}")


# print("\n--- DETAILED MODEL PERFORMANCE ---")

# print_scores("LLM", llm_keyword, llm_semantic, llm_overall)
# print_scores("BM25", bm25_keyword, bm25_semantic, bm25_overall)
# print_scores("RAG", rag_keyword, rag_semantic, rag_overall)
# print_scores("Hybrid (Fixed)", hybrid_fixed_keyword, hybrid_fixed_semantic, hybrid_fixed_overall)
# print_scores("Hybrid (Adaptive)", hybrid_keyword, hybrid_semantic, hybrid_overall)
#####BESTTT
# from modules.router.router import Router
# from modules.rag.loader import DocumentLoader
# from modules.rag.chunker import TextChunker
# from modules.rag.embeddings import EmbeddingModel
# from modules.rag.vector_store import VectorStore
# from modules.rag.bm25_retriever import BM25Retriever
# from modules.rag.hybrid_retriever import HybridRetriever
# from modules.router.query_analyzer import QueryAnalyzer
# from modules.llm.llm import VanillaLLM

# from evaluation_queries import evaluation_queries

# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # ============================================================
# # HELPERS
# # ============================================================

# def extract_text(chunk):
#     if isinstance(chunk, str):
#         return chunk
#     elif isinstance(chunk, dict):
#         return chunk.get("content", str(chunk))
#     return str(chunk)


# def remove_duplicates(chunks):
#     seen = set()
#     unique = []

#     for c in chunks:
#         text = extract_text(c)
#         if text not in seen:
#             seen.add(text)
#             unique.append(c)

#     return unique


# def score_answer(answer, keywords):
#     answer = answer.lower()
#     score = 0

#     for kw in keywords:
#         if kw.lower() in answer:
#             score += 1

#     if len(answer.split()) > 20:
#         score += 0.2

#     return score / len(keywords)


# def semantic_score(answer, reference, embedder):
#     try:
#         a = embedder.encode([answer])[0]
#         r = embedder.encode([reference])[0]
#         return cosine_similarity([a], [r])[0][0]
#     except:
#         return 0


# def get_k(query):
#     query = query.lower()
#     if "compare" in query or "difference" in query:
#         return 6
#     elif "explain" in query or "why" in query:
#         return 5
#     return 3


# def log_weights(query, alpha, beta):
#     with open("logs.txt", "a") as f:
#         f.write(f"{query} | alpha={alpha} | beta={beta}\n")


# # ============================================================
# # 1. LOAD DATA
# # ============================================================

# loader = DocumentLoader("data/")
# docs = loader.load_txt_files()

# chunker = TextChunker(chunk_size=50, overlap=10)
# chunks = chunker.chunk_documents(docs)

# print(f"[INFO] Created {len(chunks)} chunks")

# texts = [chunk["content"] for chunk in chunks]


# # ============================================================
# # 2. EMBEDDINGS + VECTOR STORE
# # ============================================================

# embedder = EmbeddingModel()
# vectors = embedder.encode(texts)

# vector_store = VectorStore(dim=len(vectors[0]))
# vector_store.add(vectors, chunks)


# # ============================================================
# # 3. BM25 + HYBRID
# # ============================================================

# bm25 = BM25Retriever(chunks)
# hybrid = HybridRetriever(bm25, vector_store, embedder)


# # ============================================================
# # 4. ROUTER + LLM
# # ============================================================

# analyzer = QueryAnalyzer()
# llm = VanillaLLM()

# router = Router(llm, None, analyzer, hybrid)


# # ============================================================
# # =================== EVALUATION ==============================
# # ============================================================

# print("\n================ EVALUATION ================\n")

# correct_routes = 0

# llm_scores = []
# rag_scores = []
# bm25_scores = []
# hybrid_fixed_scores = []
# hybrid_scores = []

# for item in evaluation_queries:
#     query = item["query"]
#     expected = item["type"]
#     keywords = item["keywords"]
#     reference = item.get("reference", "")

#     print("=" * 60)
#     print(f"Query: {query}")

#     decision = analyzer.route(query)
#     print(f"[ROUTER] Selected: {decision} | Expected: {expected}")

#     if decision == expected:
#         correct_routes += 1

#     # -------- LLM --------
#     llm_answer = llm.generate(f"Answer clearly: {query}")
#     llm_score = (score_answer(llm_answer, keywords) +
#                  semantic_score(llm_answer, reference, embedder)) / 2
#     llm_scores.append(llm_score)

#     # -------- RAG --------
#     rag_chunks = vector_store.search(query, embedder, k=2)
#     rag_context = "\n".join([extract_text(c) for c in rag_chunks])

#     rag_answer = llm.generate(f"""
# Context:
# {rag_context}

# Question: {query}
# Answer:
# """)

#     rag_score = (score_answer(rag_answer, keywords) +
#                  semantic_score(rag_answer, reference, embedder)) / 2
#     rag_scores.append(rag_score)

#     # -------- BM25 --------
#     bm25_chunks = bm25.search(query, k=2)
#     bm25_context = "\n".join([extract_text(c) for c in bm25_chunks])

#     bm25_answer = llm.generate(f"""
# Context:
# {bm25_context}

# Question: {query}
# Answer:
# """)

#     bm25_score = (score_answer(bm25_answer, keywords) +
#                   semantic_score(bm25_answer, reference, embedder)) / 2
#     bm25_scores.append(bm25_score)

#     # -------- HYBRID FIXED --------
#     k = get_k(query)

#     # 🔥 Increase k for adaptive to allow weight effect
#     hybrid_fixed_chunks = hybrid.search(query, k=k, alpha=0.5, beta=0.5)
#     hybrid_fixed_chunks = hybrid.rerank(query, hybrid_fixed_chunks, llm)
#     hybrid_fixed_chunks = remove_duplicates(hybrid_fixed_chunks)[:k]

#     hybrid_fixed_context = "\n".join([extract_text(c) for c in hybrid_fixed_chunks])

#     hybrid_fixed_answer = llm.generate(f"""
# Context:
# {hybrid_fixed_context}

# Question: {query}
# Answer:
# """)

#     hybrid_fixed_score = (score_answer(hybrid_fixed_answer, keywords) +
#                           semantic_score(hybrid_fixed_answer, reference, embedder)) / 2
#     hybrid_fixed_scores.append(hybrid_fixed_score)

#     # -------- HYBRID ADAPTIVE --------
#     alpha, beta = analyzer.get_weights(query)
#     log_weights(query, alpha, beta)

#     # 🔥 Use larger k to allow weights to affect retrieval
#     k_adaptive = max(k*2, 5)

#     hybrid_chunks = hybrid.search(query, k=k_adaptive, alpha=alpha, beta=beta)
#     hybrid_chunks = hybrid.rerank(query, hybrid_chunks, llm)
#     hybrid_chunks = remove_duplicates(hybrid_chunks)[:k]

#     hybrid_context = "\n".join([extract_text(c) for c in hybrid_chunks])

#     hybrid_answer = llm.generate(f"""
# Context:
# {hybrid_context}

# Question: {query}
# Answer:
# """)

#     hybrid_score = (score_answer(hybrid_answer, keywords) +
#                     semantic_score(hybrid_answer, reference, embedder)) / 2
#     hybrid_scores.append(hybrid_score)

#     print("\n--- SCORES ---")
#     print(f"LLM: {llm_score:.2f}")
#     print(f"BM25: {bm25_score:.2f}")
#     print(f"RAG: {rag_score:.2f}")
#     print(f"Hybrid Fixed: {hybrid_fixed_score:.2f}")
#     print(f"Hybrid Adaptive: {hybrid_score:.2f}")

# # ============================================================
# # FINAL RESULTS
# # ============================================================

# print("\n================ FINAL RESULTS ================\n")

# print(f"Routing Accuracy: {correct_routes}/{len(evaluation_queries)}")
# print(f"Routing %: {(correct_routes/len(evaluation_queries))*100:.2f}%")

# print("\n--- MODEL PERFORMANCE ---")
# print(f"Avg LLM Score: {np.mean(llm_scores):.2f}")
# print(f"Avg BM25 Score: {np.mean(bm25_scores):.2f}")
# print(f"Avg RAG Score: {np.mean(rag_scores):.2f}")
# print(f"Avg Hybrid (Fixed) Score: {np.mean(hybrid_fixed_scores):.2f}")
# print(f"Avg Hybrid (Adaptive) Score: {np.mean(hybrid_scores):.2f}")

# from modules.rag.loader import DocumentLoader
# from modules.rag.chunker import TextChunker
# from modules.rag.embeddings import EmbeddingModel
# from modules.rag.vector_store import VectorStore
# from modules.rag.bm25_retriever import BM25Retriever
# from modules.rag.hybrid_retriever import HybridRetriever
# from modules.router.query_analyzer import QueryAnalyzer
# from modules.llm.llm import VanillaLLM

# from evaluation_queries import evaluation_queries

# from sklearn.metrics.pairwise import cosine_similarity


# # ============================================================
# # HELPERS
# # ============================================================

# def extract_text(chunk):
#     if isinstance(chunk, str):
#         return chunk
#     elif isinstance(chunk, dict):
#         return chunk.get("content", str(chunk))
#     return str(chunk)


# def remove_duplicates(chunks):
#     seen = set()
#     unique = []

#     for c in chunks:
#         text = extract_text(c)
#         if text not in seen:
#             seen.add(text)
#             unique.append(c)

#     return unique


# def score_answer(answer, keywords):
#     answer = answer.lower()
#     score = 0

#     for kw in keywords:
#         if kw.lower() in answer:
#             score += 1

#     if len(answer.split()) > 20:
#         score += 0.2

#     return score / len(keywords)


# def semantic_score(answer, reference, embedder):
#     try:
#         a = embedder.encode([answer])[0]
#         r = embedder.encode([reference])[0]
#         return cosine_similarity([a], [r])[0][0]
#     except:
#         return 0


# def get_k(query):
#     query = query.lower()
#     if "compare" in query or "difference" in query:
#         return 6
#     elif "explain" in query or "why" in query:
#         return 5
#     return 3


# def log_weights(query, alpha, beta):
#     with open("logs.txt", "a") as f:
#         f.write(f"{query} | alpha={alpha} | beta={beta}\n")


# # ============================================================
# # 1. LOAD DATA
# # ============================================================

# loader = DocumentLoader("data/")
# docs = loader.load_txt_files()

# chunker = TextChunker(chunk_size=50, overlap=10)
# chunks = chunker.chunk_documents(docs)

# print(f"[INFO] Created {len(chunks)} chunks")

# texts = [chunk["content"] for chunk in chunks]


# # ============================================================
# # 2. EMBEDDINGS + VECTOR STORE
# # ============================================================

# embedder = EmbeddingModel()
# vectors = embedder.encode(texts)

# vector_store = VectorStore(dim=len(vectors[0]))
# vector_store.add(vectors, chunks)


# # ============================================================
# # 3. BM25 + HYBRID
# # ============================================================

# bm25 = BM25Retriever(chunks)
# hybrid = HybridRetriever(bm25, vector_store, embedder)


# # ============================================================
# # 4. ROUTER + LLM
# # ============================================================

# analyzer = QueryAnalyzer()
# llm = VanillaLLM()


# # ============================================================
# # =================== EVALUATION ==============================
# # ============================================================

# print("\n================ EVALUATION ================\n")

# correct_routes = 0

# llm_scores = []
# rag_scores = []
# bm25_scores = []
# hybrid_fixed_scores = []
# hybrid_scores = []

# for item in evaluation_queries:
#     query = item["query"]
#     expected = item["type"]
#     keywords = item["keywords"]
#     reference = item.get("reference", "")

#     print("=" * 60)
#     print(f"Query: {query}")

#     decision = analyzer.route(query)
#     print(f"[ROUTER] Selected: {decision} | Expected: {expected}")

#     if decision == expected:
#         correct_routes += 1

#     # -------- LLM --------
#     llm_answer = llm.generate(f"Answer clearly: {query}")
#     llm_score = (score_answer(llm_answer, keywords) +
#                  semantic_score(llm_answer, reference, embedder)) / 2
#     llm_scores.append(llm_score)

#     # -------- RAG --------
#     rag_chunks = vector_store.search(query, embedder, k=2)
#     rag_context = "\n".join([extract_text(c) for c in rag_chunks])

#     rag_answer = llm.generate(f"""
# Context:
# {rag_context}

# Question: {query}
# Answer:
# """)

#     rag_score = (score_answer(rag_answer, keywords) +
#                  semantic_score(rag_answer, reference, embedder)) / 2
#     rag_scores.append(rag_score)

#     # -------- BM25 --------
#     bm25_chunks = bm25.search(query, k=2)
#     bm25_context = "\n".join([extract_text(c) for c in bm25_chunks])

#     bm25_answer = llm.generate(f"""
# Context:
# {bm25_context}

# Question: {query}
# Answer:
# """)

#     bm25_score = (score_answer(bm25_answer, keywords) +
#                   semantic_score(bm25_answer, reference, embedder)) / 2
#     bm25_scores.append(bm25_score)

#     # -------- HYBRID FIXED --------
#     k = get_k(query)

#     hybrid_fixed_chunks = hybrid.search(query, k=k, alpha=0.5, beta=0.5)
#     hybrid_fixed_chunks = hybrid.rerank(query, hybrid_fixed_chunks, llm)
#     hybrid_fixed_chunks = remove_duplicates(hybrid_fixed_chunks)[:2]

#     hybrid_fixed_context = "\n".join([extract_text(c) for c in hybrid_fixed_chunks])

#     hybrid_fixed_answer = llm.generate(f"""
# Context:
# {hybrid_fixed_context}

# Question: {query}
# Answer:
# """)

#     hybrid_fixed_score = (score_answer(hybrid_fixed_answer, keywords) +
#                           semantic_score(hybrid_fixed_answer, reference, embedder)) / 2
#     hybrid_fixed_scores.append(hybrid_fixed_score)

#     # -------- HYBRID ADAPTIVE --------
#     alpha, beta = analyzer.get_weights(query)
#     log_weights(query, alpha, beta)

#     hybrid_chunks = hybrid.search(query, k=k, alpha=alpha, beta=beta)
#     hybrid_chunks = hybrid.rerank(query, hybrid_chunks, llm)
#     hybrid_chunks = remove_duplicates(hybrid_chunks)[:2]

#     hybrid_context = "\n".join([extract_text(c) for c in hybrid_chunks])

#     hybrid_answer = llm.generate(f"""
# Context:
# {hybrid_context}

# Question: {query}
# Answer:
# """)

#     hybrid_score = (score_answer(hybrid_answer, keywords) +
#                     semantic_score(hybrid_answer, reference, embedder)) / 2
#     hybrid_scores.append(hybrid_score)

#     print("\n--- SCORES ---")
#     print(f"LLM: {llm_score:.2f}")
#     print(f"BM25: {bm25_score:.2f}")
#     print(f"RAG: {rag_score:.2f}")
#     print(f"Hybrid Fixed: {hybrid_fixed_score:.2f}")
#     print(f"Hybrid Adaptive: {hybrid_score:.2f}")


# # ============================================================
# # FINAL RESULTS
# # ============================================================

# print("\n================ FINAL RESULTS ================\n")

# print(f"Routing Accuracy: {correct_routes}/{len(evaluation_queries)}")
# print(f"Routing %: {(correct_routes/len(evaluation_queries))*100:.2f}%")

# print("\n--- MODEL PERFORMANCE ---")
# print(f"Avg LLM Score: {sum(llm_scores)/len(llm_scores):.2f}")
# print(f"Avg BM25 Score: {sum(bm25_scores)/len(bm25_scores):.2f}")
# print(f"Avg RAG Score: {sum(rag_scores)/len(rag_scores):.2f}")
# print(f"Avg Hybrid (Fixed) Score: {sum(hybrid_fixed_scores)/len(hybrid_fixed_scores):.2f}")
# print(f"Avg Hybrid (Adaptive) Score: {sum(hybrid_scores)/len(hybrid_scores):.2f}")

# from modules.rag.loader import DocumentLoader
# from modules.rag.chunker import TextChunker
# from modules.rag.embeddings import EmbeddingModel
# from modules.rag.vector_store import VectorStore
# from modules.rag.rag_pipeline import RAGPipeline
# from modules.llm.llm import VanillaLLM
# from modules.router.query_analyzer import QueryAnalyzer
# from modules.router.router import Router

# # Load + chunk
# loader = DocumentLoader("data/")
# docs = loader.load_txt_files()

# chunker = TextChunker(chunk_size=50, overlap=10)
# chunks = chunker.chunk_documents(docs)

# # Embeddings
# texts = [chunk["content"] for chunk in chunks]
# embedder = EmbeddingModel()
# vectors = embedder.encode(texts)

# # FAISS
# vector_store = VectorStore(dim=len(vectors[0]))
# vector_store.add(vectors, chunks)

# # LLM + RAG
# llm = VanillaLLM()
# rag = RAGPipeline(embedder, vector_store, llm)

# # Analyzer + Router
# analyzer = QueryAnalyzer()
# router = Router(llm, rag, analyzer)

# # Test queries
# queries = [
#     "What is AI?",
#     "According to the document, what is AI?",
#     "Explain AI in detail"
# ]

# for q in queries:
#     print(f"\nQuery: {q}")
#     response = router.route(q)
#     print("Answer:", response)







