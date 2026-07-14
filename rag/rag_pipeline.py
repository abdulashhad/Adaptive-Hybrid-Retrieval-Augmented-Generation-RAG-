class RAGPipeline:
    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def generate(self, query, k=2):
        # Step 1: Embed query
        query_vector = self.embedder.encode([query])[0]

        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query_vector, k)

        # Step 3: Build context
        context = "\n".join([chunk["content"] for chunk in retrieved_chunks])

        # Step 4: Create prompt
        prompt = f"""
        You are a precise assistant.

        Answer the question ONLY using the context below.
        Give a complete and clear sentence.
        Do not omit important words.

        Context:
        {context}

        Question:
        {query}
        """
        
        # Step 5: Generate answer
        response = self.llm.generate(prompt)

        return response