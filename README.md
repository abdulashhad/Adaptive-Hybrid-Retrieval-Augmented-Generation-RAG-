# Adaptive-Hybrid-Retrieval-Augmented-Generation-RAG-
Large Language Models (LLMs) often generate hallucinated or outdated responses because they rely solely on their training data. Retrieval-Augmented Generation (RAG) addresses this issue by retrieving relevant external information before generating responses.

This project proposes an Adaptive Hybrid Retrieval Framework that combines BM25 lexical retrieval and FAISS semantic retrieval with a query-aware routing mechanism. Instead of using a single retrieval strategy for every query, the system dynamically selects and weights retrieval methods based on the query characteristics, improving retrieval quality and response accuracy.

**Key Features**
Vanilla LLM baseline
BM25 lexical retrieval
FAISS semantic retrieval
Hybrid BM25 + FAISS retrieval
Query-aware adaptive retrieval
Dynamic retrieval weighting
LLM-based re-ranking
Performance evaluation using keyword and semantic similarity metrics

**System Architecture**

**Workflow**

| Component            | Technology                          |
| -------------------- | ----------------------------------- |
| Programming Language | Python                              |
| LLM                  | FLAN-T5 Large                       |
| Semantic Retrieval   | FAISS                               |
| Lexical Retrieval    | BM25                                |
| Embeddings           | Sentence Transformers               |
| Evaluation           | Keyword Match + Semantic Similarity |

**Experimental Results**
The proposed Adaptive Hybrid RAG consistently outperformed standalone LLM, BM25, Standard RAG, and Fixed Hybrid RAG across all evaluation metrics.

Performance Comparison

| Model                   |  Keyword | Semantic |  Overall |
| ----------------------- | -------: | -------: | -------: |
| LLM (Baseline)          |     0.48 |     0.67 |     0.58 |
| BM25                    |     0.80 |     0.64 |     0.72 |
| Standard RAG            |     0.68 |     0.71 |     0.69 |
| Hybrid RAG              |     0.92 |     0.71 |     0.81 |
| **Adaptive Hybrid RAG** | **0.94** | **0.72** | **0.83** |

Top-k Evaluation

| Top-k | BM25 | Standard RAG | Hybrid | Adaptive Hybrid |
| ----: | ---: | -----------: | -----: | --------------: |
|     1 | 0.46 |         0.63 |   0.60 |        **0.65** |
|     3 | 0.68 |         0.70 |   0.74 |        **0.80** |
|     5 | 0.72 |         0.71 |   0.80 |        **0.80** |
|     7 | 0.72 |         0.69 |   0.81 |        **0.83** |


**Query Routing Performance**
The rule-based query analyzer achieved **84% routing accuracy**, enabling dynamic selection between LLM-only, Standard RAG, and Adaptive Hybrid RAG based on query characteristics.

**Future Improvements**
GraphRAG integration
Agentic retrieval framework
Multimodal document retrieval
Learning-based query routing
Cross-encoder re-ranking

**Project Structure**
Adaptive-Hybrid-RAG
│
├── data/
├── modules/
│   ├── router/
│   ├── rag/
│   ├── llm/
│
├── docs/
├── main.py
├── requirements.txt
└── README.md
