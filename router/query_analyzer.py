class QueryAnalyzer:

    # ---------------------------
    # FEATURE EXTRACTION
    # ---------------------------
    def analyze(self, query):
        query = query.lower()
        words = query.split()

        return {
            "length": len(words),
            "is_short": len(words) <= 3,

            "has_question": any(w in query for w in [
                "what", "define", "who", "when"
            ]),

            "is_explanatory": any(w in query for w in [
                "explain", "why", "how"
            ]),

            "is_comparison": any(w in query for w in [
                "compare", "difference", "vs"
            ]),

            # detects technical depth
            "has_domain_terms": any(w in query for w in [
                "neural", "learning", "model", "algorithm",
                "network", "training", "data"
            ])
        }

    # ---------------------------
    # ROUTING LOGIC (IMPROVED)
    # ---------------------------

    # def route(self, query):
    #     f = self.analyze(query)

    #     # RULE 1: SHORT SIMPLE → LLM
    #     if f["is_short"] and not f["is_explanatory"]:
    #         return "LLM"

    #     # RULE 2: COMPARISON → HYBRID
    #     if f["is_comparison"]:
    #         return "HYBRID"

    #     # RULE 3: FACTUAL QUESTION → LLM
    #     if f["has_question"] and f["length"] <= 5:
    #         return "LLM"

    #     # RULE 4: EXPLANATORY QUERIES
    #     if f["is_explanatory"]:
    #         if f["length"] > 5 or f["has_domain_terms"]:
    #             return "HYBRID"   # complex explanation
    #         else:
    #             return "RAG"      # simple explanation

    #     # RULE 5: DEFAULT → RAG
    #     return "RAG"

    def route(self, query):
        f = self.analyze(query)
        q = query.lower()
        words = q.split()

        # ---------------------------
        # 1. SHORT → LLM
        # ---------------------------
        if f["is_short"]:
            return "LLM"

        # ---------------------------
        # 2. SIMPLE FACTUAL → LLM
        # ---------------------------
        if f["has_question"] and f["length"] <= 8 and not f["has_domain_terms"]:
            return "LLM"

        # ---------------------------
        # 3. COMPARISON → HYBRID
        # ---------------------------
        if f["is_comparison"]:
            return "HYBRID"

        # ---------------------------
        # 4. EXPLANATION LOGIC 🔥 (FIXED)
        # ---------------------------
        if f["is_explanatory"]:

            # multi-concept detection
            multi_concept = (
                " and " in q or
                " vs " in q or
                " between " in q or
                "better" in q or
                "difference" in q or
                f["length"] >= 5
            )

            if multi_concept:
                return "HYBRID"   # 🔥 THIS FIXES YOUR MAIN ISSUE

            return "RAG"

        # ---------------------------
        # 5. DOMAIN HEAVY → HYBRID
        # ---------------------------
        if f["has_domain_terms"] and f["length"] > 5:
            return "HYBRID"

        # ---------------------------
        # DEFAULT
        # ---------------------------
        return "RAG"

#GIVING GOOD 03/05/2006
    # def route(self, query):
    #     f = self.analyze(query)

    #     # SHORT FACTUAL → LLM
    #     if f["is_short"]:
    #         return "LLM"

    #     # SIMPLE WHAT/DEFINE → LLM
    #     if f["has_question"] and f["length"] <= 6:
    #         return "LLM"

    #     # COMPARISON → HYBRID
    #     if f["is_comparison"]:
    #         return "HYBRID"

    #     # EXPLANATION
    #     if f["is_explanatory"]:

    #         # multi-concept → hybrid
    #         if "and" in query or "with" in query:
    #             return "HYBRID"

    #         # otherwise semantic
    #         return "RAG"

    #     # DEFAULT
    #     return "RAG"
    
    # def get_weights(self, query):
    #     q = query.lower()
    #     length = len(q.split())

    #     # FACTUAL → STRONG BM25
    #     if "what is" in q or "define" in q:
    #         return 0.9, 0.1

    #     # VERY SHORT → BM25 DOMINANT
    #     if length <= 3:
    #         return 0.85, 0.15

    #     # EXPLANATION → STRONG SEMANTIC
    #     if "explain" in q or "why" in q or "how" in q:
    #         return 0.1, 0.9

    #     # COMPARISON → TRUE HYBRID
    #     if "compare" in q or "difference" in q or "vs" in q:
    #         return 0.5, 0.5

    #     # DEFAULT
    #     return 0.3, 0.7

# OKAYISH 03/01/2026
    # def get_weights(self, query):
    #     q = query.lower()
    #     length = len(q.split())

    #     # FACTUAL → keyword leaning
    #     if "what is" in q or "define" in q:
    #         return 0.75, 0.25   # (alpha = BM25, beta = FAISS)

    #     # SHORT → keyword leaning
    #     if length <= 3:
    #         return 0.7, 0.3

    #     # EXPLANATION → semantic leaning (balanced, not extreme)
    #     if "explain" in q or "why" in q or "how" in q:
    #         return 0.4, 0.6

    #     # COMPARISON → balanced
    #     if "compare" in q or "difference" in q or "vs" in q:
    #         return 0.5, 0.5

    #     # DEFAULT
    #     return 0.45, 0.55


    def get_weights(self, query):
        q = query.lower()
        length = len(q.split())

        # FACTUAL → STRONG BM25
        if "what is" in q or "define" in q:
            return 0.85, 0.15

        # VERY SHORT → BM25 dominant
        if length <= 3:
            return 0.8, 0.2

        # COMPARISON → TRUE HYBRID
        if "compare" in q or "difference" in q or "vs" in q:
            return 0.5, 0.5

        # EXPLANATION (KEY FIX 🔥)
        if "explain" in q or "why" in q or "how" in q:

            # complex explanation → strong semantic
            if length >= 5:
                return 0.25, 0.75   # 🔥 push FAISS HARD

            return 0.35, 0.65

        # DEFAULT
        return 0.4, 0.6

    #GOODDD
    # ---------------------------
    # ADAPTIVE WEIGHTING (IMPROVED)
    # ---------------------------
    # def get_weights(self, query):
    #     q = query.lower()
    #     length = len(q.split())

    #     # COMPARISON → BALANCED
    #     if "compare" in q or "difference" in q or "vs" in q:
    #         return 0.5, 0.5   # (BM25, Semantic)

    #     # FACTUAL → KEYWORD HEAVY
    #     if "what is" in q or "define" in q:
    #         return 0.8, 0.2

    #     # VERY SHORT → KEYWORD DOMINANT
    #     if length <= 3:
    #         return 0.75, 0.25

    #     # EXPLANATION → SEMANTIC HEAVY
    #     if "explain" in q or "why" in q or "how" in q:
    #         return 0.2, 0.8

    #     # DEFAULT → SLIGHT SEMANTIC BIAS
    #     return 0.4, 0.6

#BEST
# class QueryAnalyzer:

#     def analyze(self, query):
#         query = query.lower()
#         words = query.split()

#         return {
#             "length": len(words),
#             "is_short": len(words) <= 3,
#             "has_question": any(w in query for w in ["what", "define"]),
#             "is_explanatory": any(w in query for w in ["explain", "why", "how"]),
#             "is_comparison": any(w in query for w in ["compare", "difference"]),
#         }

#     # ---------------------------
#     # ROUTING (FIXED 🔥)
#     # ---------------------------
#     def route(self, query):
#         features = self.analyze(query)

#         score = 0

#         # LONG QUERY
#         if features["length"] > 4:
#             score += 1

#         # EXPLANATION → IMPORTANT
#         if features["is_explanatory"]:
#             score += 2   # 👈 BOOST

#         # COMPARISON → IMPORTANT
#         if features["is_comparison"]:
#             score += 2   # 👈 BOOST

#         # ---------------------------
#         # DECISION
#         # ---------------------------
#         if score <= 1:
#             return "LLM"
#         elif score == 2:
#             return "RAG"
#         else:
#             return "HYBRID"

#     # ---------------------------
#     # WEIGHTS (unchanged)
#     # ---------------------------
#     def get_weights(self, query):
#         query = query.lower()

#         if "compare" in query or "difference" in query:
#             return 0.5, 0.5   # balanced

#         elif "what is" in query or "define" in query:
#             return 0.7, 0.3   # BM25 heavy (keywords matter)

#         elif "explain" in query or "why" in query:
#             return 0.3, 0.7   # semantic heavy

#         else:
#             return 0.5, 0.5
    # def get_weights(self, query):
    #     query = query.lower()

    #     if "what is" in query:
    #         return 0.7, 0.3   # BM25 more

    #     elif "explain" in query or "why" in query:
    #         return 0.3, 0.7   # semantic more

    #     elif "compare" in query or "difference" in query:
    #         return 0.5, 0.5   # balanced

    #     return 0.5, 0.5

    # def get_weights(self, query):
    #     features = self.analyze(query)

    #     alpha = 0.5
    #     beta = 0.5

    #     if features["is_short"]:
    #         alpha = 0.7
    #         beta = 0.3

    #     elif features["is_explanatory"]:
    #         alpha = 0.3
    #         beta = 0.7

    #     elif features["is_comparison"]:
    #         alpha = 0.5
    #         beta = 0.5

    #     return alpha, beta

# class QueryAnalyzer:
#     def analyze(self, query):
#         query = query.lower()
#         words = query.split()

#         score = {
#             "is_short": len(words) <= 3,
#             "has_question": any(w in query for w in ["what", "define"]),
#             "is_explanatory": any(w in query for w in ["explain", "why", "how"]),
#             "is_comparison": "compare" in query or "difference" in query,
#         }

#         return score


#     def get_weights(self, query):
#         features = self.analyze(query)

#         # Default
#         alpha = 0.5  # BM25
#         beta = 0.5   # FAISS

#         # 🔥 RULES

#         # Short factual queries → BM25
#         if features["is_short"] or features["has_question"]:
#             alpha = 0.7
#             beta = 0.3

#         # Explanation queries → FAISS
#         elif features["is_explanatory"]:
#             alpha = 0.3
#             beta = 0.7

#         # Comparison → balanced
#         elif features["is_comparison"]:
#             alpha = 0.5
#             beta = 0.5

#         return alpha, beta

# class QueryAnalyzer:

#     # ---------------------------
#     # FEATURE EXTRACTION
#     # ---------------------------
#     def analyze(self, query):
#         query = query.lower()
#         words = query.split()

#         return {
#             "length": len(words),
#             "is_short": len(words) <= 3,
#             "has_question": any(w in query for w in ["what", "define"]),
#             "is_explanatory": any(w in query for w in ["explain", "why", "how"]),
#             "is_comparison": any(w in query for w in ["compare", "difference"])
#         }


#     # ---------------------------
#     # DYNAMIC WEIGHTING (HYBRID)
#     # ---------------------------
#     def get_weights(self, query):
#         features = self.analyze(query)

#         alpha = 0.5  # BM25
#         beta = 0.5   # FAISS

#         if features["is_short"] or features["has_question"]:
#             alpha = 0.7
#             beta = 0.3

#         elif features["is_explanatory"]:
#             alpha = 0.3
#             beta = 0.7

#         elif features["is_comparison"]:
#             alpha = 0.5
#             beta = 0.5

#         return alpha, beta


#     # ---------------------------
#     # ADAPTIVE ROUTING (FINAL 🔥)
#     # ---------------------------
#     def route(self, query):
#         features = self.analyze(query)

#         score = 0

#         if features["length"] > 5:
#             score += 1

#         if features["is_explanatory"]:
#             score += 1

#         if features["is_comparison"]:
#             score += 1

#         # ---------------------------
#         # DECISION
#         # ---------------------------
#         if score <= 1:
#             return "LLM"
#         elif score == 2:
#             return "RAG"
#         else:
#             return "HYBRID"