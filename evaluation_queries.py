
evaluation_queries = [

    # {
    #     "query": "Explain machine learning algorithms",
    #     "type": "HYBRID",
    #     "keywords": ["machine learning", "algorithms", "models"],
    #     "reference": "Machine learning algorithms are methods used to learn patterns from data and make predictions."
    # },

    {
        "query": "What is AI?",
        "type": "LLM",
        "keywords": ["intelligence", "machines"],
        "reference": "Artificial Intelligence refers to machines that simulate human intelligence."
    },
    {
        "query": "Explain AI in healthcare",
        "type": "RAG",
        "keywords": ["healthcare", "diagnosis", "treatment"],
        "reference": "AI in healthcare is used for diagnosis, treatment planning, and improving medical outcomes."
    },
    
    {
        "query": "Define supervised learning",
        "type": "LLM",
        "keywords": ["supervised", "labeled data", "training"],
        "reference": "Supervised learning is a machine learning approach where models are trained using labeled data."
    },

    {
        "query": "Explain machine learning algorithms and compare their performance in real-world applications",
        "type": "HYBRID",
        "keywords": ["machine learning", "algorithms", "performance", "applications"],
        "reference": "Machine learning algorithms learn from data and their performance varies across real-world applications based on data and model design."
    },

    {
        "query": "What is artificial intelligence?",
        "type": "LLM",
        "keywords": ["intelligence", "machines"],
        "reference": "Artificial intelligence refers to machines that simulate human intelligence."
    },

    # {
    #     "query": "Explain how neural networks are used in deep learning systems",
    #     "type": "RAG",
    #     "keywords": ["neural networks", "deep learning", "layers", "training"],
    #     "reference": "Neural networks are used in deep learning to model complex patterns through layers of interconnected neurons."
    # },

    {
        "query": "Compare supervised and unsupervised learning with examples and applications",
        "type": "HYBRID",
        "keywords": ["supervised", "unsupervised", "examples", "applications"],
        "reference": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data with different applications."
    },

    # {
    #     "query": "Why is feature engineering important and how does it impact model performance?",
    #     "type": "HYBRID",
    #     "keywords": ["feature engineering", "model performance", "data", "features"],
    #     "reference": "Feature engineering improves model performance by selecting and transforming relevant input variables."
    # },

    {
        "query": "Explain the vanishing gradient problem in deep neural networks",
        "type": "RAG",
        "keywords": ["vanishing gradient", "deep networks", "training", "backpropagation"],
        "reference": "The vanishing gradient problem occurs when gradients become too small during backpropagation, slowing learning."
    },

    # {
    #     "query": "What is overfitting in machine learning?",
    #     "type": "LLM",
    #     "keywords": ["overfitting", "model", "training"],
    #     "reference": "Overfitting occurs when a model learns training data too well and fails to generalize to new data."
    # },

    {
        "query": "Explain gradient descent and compare it with stochastic gradient descent",
        "type": "HYBRID",
        "keywords": ["gradient descent", "stochastic", "optimization", "learning"],
        "reference": "Gradient descent updates weights using full data, while stochastic gradient descent uses single samples for faster updates."
    },

    {
        "query": "How does transfer learning work and why is it useful in deep learning?",
        "type": "HYBRID",
        "keywords": ["transfer learning", "deep learning", "pretrained", "models"],
        "reference": "Transfer learning reuses pretrained models to improve performance and reduce training time."
    },

    {
        "query": "Explain the role of activation functions in neural networks",
        "type": "RAG",
        "keywords": ["activation function", "neural networks", "non-linearity"],
        "reference": "Activation functions introduce non-linearity in neural networks, enabling them to learn complex patterns."
    },

    {
        "query": "Difference between CNN and RNN in terms of architecture and use cases",
        "type": "HYBRID",
        "keywords": ["cnn", "rnn", "architecture", "use cases"],
        "reference": "CNNs are used for spatial data like images, while RNNs handle sequential data like time series."
    },

    {
        "query": "Why is data preprocessing necessary before training machine learning models?",
        "type": "HYBRID",
        "keywords": ["data preprocessing", "cleaning", "normalization", "training"],
        "reference": "Data preprocessing improves data quality and ensures better model training and performance."
    }
]

# for item in evaluation_queries:
#     print("=" * 40)
#     print("Query:", item["query"])
#     print("Type :", item["type"])
#     print("=" * 40)
#     print()

# evaluation_queries += [
#     {
#         "query": "Compare machine learning and deep learning",
#         "type": "HYBRID",
#         "keywords": ["subset", "neural networks", "learning"],
#         "reference": "Machine learning is a subset of AI, while deep learning is a subset of machine learning using neural networks."
#     },
    
#     {
#         "query": "Explain the role of neural networks in deep learning",
#         "type": "RAG",
#         "keywords": ["neural networks", "deep learning", "layers"],
#         "reference": "Neural networks form the foundation of deep learning by modeling complex patterns through multiple layers."
#     },

#     {
#         "query": "Difference between AI and ML in healthcare",
#         "type": "HYBRID",
#         "keywords": ["AI", "ML", "healthcare", "difference"],
#         "reference": "AI is a broader concept, while ML is a subset used in healthcare for predictive modeling and data analysis."
#     },
    
#     {
#         "query": "Why is deep learning better than traditional ML?",
#         "type": "HYBRID",
#         "keywords": ["deep learning", "neural networks", "better"],
#         "reference": "Deep learning performs better due to its ability to learn complex patterns using neural networks."
#     },
        
#     {
#         "query": "Explain machine learning algorithms",
#         "type": "HYBRID",
#         "keywords": ["machine learning", "algorithms", "models"],
#         "reference": "Machine learning algorithms are methods used to learn patterns from data and make predictions."
#     }
# ]
# # ]

# evaluation_queries += [

#     {
#         "query": "Define supervised learning",
#         "type": "LLM",
#         "keywords": ["supervised", "labeled data", "training"],
#         "reference": "Supervised learning is a machine learning approach where models are trained using labeled data."
#     },

#     {
#         "query": "Explain unsupervised learning with examples",
#         "type": "RAG",
#         "keywords": ["unsupervised", "clustering", "patterns"],
#         "reference": "Unsupervised learning identifies patterns in data without labeled outputs, such as clustering."
#     },

#     {
#         "query": "What is overfitting in machine learning?",
#         "type": "LLM",
#         "keywords": ["overfitting", "training", "generalization"],
#         "reference": "Overfitting occurs when a model learns training data too well and fails to generalize to new data."
#     },

#     {
#         "query": "Compare supervised and unsupervised learning",
#         "type": "HYBRID",
#         "keywords": ["supervised", "unsupervised", "difference"],
#         "reference": "Supervised learning uses labeled data, while unsupervised learning works with unlabeled data to find patterns."
#     },

#     {
#         "query": "Explain the role of neural networks in deep learning",
#         "type": "RAG",
#         "keywords": ["neural networks", "deep learning", "layers"],
#         "reference": "Neural networks form the foundation of deep learning by modeling complex patterns through multiple layers."
#     },

#     {
#         "query": "What is a convolutional neural network?",
#         "type": "LLM",
#         "keywords": ["CNN", "images", "convolution"],
#         "reference": "A convolutional neural network is a deep learning model designed for processing image data using convolution operations."
#     },

#     {
#         "query": "Difference between CNN and RNN",
#         "type": "HYBRID",
#         "keywords": ["CNN", "RNN", "difference", "sequence"],
#         "reference": "CNNs are used for spatial data like images, while RNNs are used for sequential data like text or time series."
#     },

#     {
#         "query": "Why are activation functions used in neural networks?",
#         "type": "HYBRID",
#         "keywords": ["activation function", "non-linearity", "neural network"],
#         "reference": "Activation functions introduce non-linearity, allowing neural networks to learn complex patterns."
#     },

#     {
#         "query": "Explain gradient descent optimization",
#         "type": "RAG",
#         "keywords": ["gradient descent", "optimization", "loss"],
#         "reference": "Gradient descent is an optimization algorithm used to minimize loss by updating model parameters iteratively."
#     },

#     {
#         "query": "What is the difference between batch and stochastic gradient descent?",
#         "type": "HYBRID",
#         "keywords": ["batch", "stochastic", "gradient descent"],
#         "reference": "Batch gradient descent uses the full dataset, while stochastic gradient descent updates parameters using one sample at a time."
#     },

#     {
#         "query": "Explain the concept of feature engineering",
#         "type": "RAG",
#         "keywords": ["features", "data", "preprocessing"],
#         "reference": "Feature engineering involves transforming raw data into meaningful features to improve model performance."
#     },

#     {
#         "query": "What is transfer learning in deep learning?",
#         "type": "LLM",
#         "keywords": ["transfer learning", "pretrained", "models"],
#         "reference": "Transfer learning involves using a pretrained model and adapting it to a new task."
#     },

#     {
#         "query": "Compare classification and regression",
#         "type": "HYBRID",
#         "keywords": ["classification", "regression", "difference"],
#         "reference": "Classification predicts discrete labels, while regression predicts continuous values."
#     },

#     {
#         "query": "Why is data preprocessing important in machine learning?",
#         "type": "HYBRID",
#         "keywords": ["data preprocessing", "cleaning", "quality"],
#         "reference": "Data preprocessing improves data quality and ensures better model performance."
#     },

#     {
#         "query": "Explain the vanishing gradient problem",
#         "type": "RAG",
#         "keywords": ["vanishing gradient", "deep networks", "training"],
#         "reference": "The vanishing gradient problem occurs when gradients become very small, slowing down training in deep networks."
#     }

# ]


# #######################################
# #EVALUATION METRICS TO CHECK MODELS ACCURACY#######################
# evaluation_queries = [
#     {
#         "query": "What is AI?",
#         "type": "LLM",
#         # "keywords": ["intelligence", "machines"]
#         "keywords": ["intelligence", "machines"],
#         "reference": "Artificial Intelligence refers to machines that simulate human intelligence."
#     },
#     {
#         "query": "Explain AI in healthcare",
#         "type": "RAG",
#         "keywords": ["healthcare", "diagnosis", "treatment"]
#     },
#     {
#         "query": "Compare machine learning and deep learning",
#         "type": "HYBRID",
#         "keywords": ["subset", "neural networks", "learning"]
#     },
#     {
#         "query": "Difference between AI and ML in healthcare",
#         "type": "HYBRID",
#         "keywords": ["AI", "ML", "healthcare", "difference"]
#     },
#     {
#         "query": "Why is deep learning better than traditional ML?",
#         "type": "HYBRID",
#         "keywords": ["deep learning", "neural networks", "better"]
#     }
# ]

####################################################
#EVALUATION QUERIES TO CHECK THE ROUTING ACCURACY##################
# evaluation_queries = [

#     # SIMPLE
#     {"query": "What is AI?", "type": "LLM"},
#     {"query": "Define machine learning", "type": "LLM"},
#     {"query": "What is deep learning?", "type": "LLM"},

#     # MEDIUM
#     {"query": "Explain AI in healthcare", "type": "RAG"},
#     {"query": "Applications of AI in finance", "type": "RAG"},
#     {"query": "How is AI used in medical imaging?", "type": "RAG"},

#     # COMPLEX
#     {"query": "Compare machine learning and deep learning", "type": "HYBRID"},
#     {"query": "Difference between AI and ML in healthcare", "type": "HYBRID"},
#     {"query": "Why is deep learning better than traditional ML?", "type": "HYBRID"},
# ]



