class TextChunker:
    def __init__(self, chunk_size=100, overlap=20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents):
        chunks = []

        for doc in documents:
            words = doc["content"].split()

            start = 0
            while start < len(words):
                end = start + self.chunk_size
                chunk_words = words[start:end]

                # Skip incomplete small chunks
                if len(chunk_words) < self.chunk_size * 0.5:
                    break

                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "content": chunk_text,
                    "source": doc["source"]
                })

                start += self.chunk_size - self.overlap

        return chunks