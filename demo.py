from modules.rag.loader import DocumentLoader
from modules.rag.chunker import TextChunker
from modules.rag.embeddings import EmbeddingModel
from modules.rag.vector_store import VectorStore
from modules.rag.bm25_retriever import BM25Retriever
from modules.rag.hybrid_retriever import HybridRetriever

from modules.router.query_analyzer import QueryAnalyzer
from modules.llm.llm import VanillaLLM


print("=" * 60)
print("ADAPTIVE HYBRID RAG DEMO")
print("=" * 60)

# ==================================================
# LOAD DATA
# ==================================================

loader = DocumentLoader("data/")
docs = loader.load_txt_files()

chunker = TextChunker(
    chunk_size=40,
    overlap=5
)

chunks = chunker.chunk_documents(docs)

print(f"[INFO] Loaded {len(chunks)} chunks")

# ==================================================
# EMBEDDINGS + FAISS
# ==================================================

embedder = EmbeddingModel()

texts = [chunk["content"] for chunk in chunks]

vectors = embedder.encode(texts)

vector_store = VectorStore(
    dim=len(vectors[0])
)

vector_store.add(vectors, chunks)

# ==================================================
# BM25 + HYBRID
# ==================================================

bm25 = BM25Retriever(chunks)

hybrid = HybridRetriever(
    bm25,
    vector_store,
    embedder
)

# ==================================================
# QUERY ANALYZER + LLM
# ==================================================

analyzer = QueryAnalyzer()

llm = VanillaLLM()

print("\nSystem Ready.\n")

# ==================================================
# INTERACTIVE LOOP
# ==================================================

while True:

    query = input("\nEnter Query (or 'exit'): ")

    if query.lower() == "exit":
        break

    decision = analyzer.route(query)

    print(f"\n[ROUTER] Decision: {decision}")

    # --------------------------------------
    # LLM
    # --------------------------------------

    if decision == "LLM":

        answer = llm.generate(query)

    # --------------------------------------
    # RAG
    # --------------------------------------

    elif decision == "RAG":

        chunks_rag = vector_store.search(
            query,
            embedder,
            k=3
        )

        context = "\n".join(
            [c["content"] for c in chunks_rag]
        )

        prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

        answer = llm.generate(prompt)

    # --------------------------------------
    # HYBRID
    # --------------------------------------

    else:

        alpha, beta = analyzer.get_weights(query)

        print(
            f"[WEIGHTS] "
            f"BM25={alpha:.2f}, "
            f"FAISS={beta:.2f}"
        )

        retrieved = hybrid.search(
            query,
            k=3,
            alpha=alpha,
            beta=beta
        )

        retrieved = hybrid.rerank(
            query,
            retrieved,
            llm
        )

        context = "\n".join(
            [c["content"] for c in retrieved]
        )

        prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

        answer = llm.generate(prompt)

    print("\nANSWER:")
    print("-" * 60)
    print(answer)
    print("-" * 60)