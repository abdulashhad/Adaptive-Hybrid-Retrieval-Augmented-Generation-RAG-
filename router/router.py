class Router:
    def __init__(self, llm, rag_pipeline, analyzer,hybrid_retriever):
        self.llm = llm
        self.rag = rag_pipeline
        self.analyzer = analyzer
        self.hybrid_retriever = hybrid_retriever

    def route(self, query):
        decision = self.analyzer.route(query)

        print(f"[ROUTER] Decision: {decision}")

        if decision == "LLM":
            return self.llm.generate(query)

        elif decision == "RAG":
            return self.rag.generate(query)

        elif decision == "HYBRID":
            return self.hybrid_response(query)

        else:
            return "Unknown decision"

def hybrid_response(self, query):
    # # Step 1: Get query embedding
    # query_vector = self.rag.embedder.encode([query])[0]

    # # Step 2: Retrieve chunks directly
    # retrieved_chunks = self.rag.vector_store.search(query_vector, k=2)

    # Step 1: Get dynamic weights (optional but powerful 🔥)
    alpha, beta = self.analyzer.get_weights(query)

    # Step 2: Use YOUR HybridRetriever
    retrieved_chunks = self.hybrid_retriever.search(
        query,
        k=3,
        alpha=alpha,
        beta=beta
    )

    # Step 3: Remove duplicate chunks
    unique_chunks = list({chunk["content"] for chunk in retrieved_chunks})

    # Step 4: Build clean context
    context = "\n".join(unique_chunks)

    # Step 5: Ask LLM to generate structured answer
    prompt = f"""
    Using the context below, give a detailed and well-structured answer.

    Context:
    {context}

    Question:
    {query}
    """

    # Step 6: Generate final answer
    final_answer = self.llm.generate(prompt)

    return final_answer
    