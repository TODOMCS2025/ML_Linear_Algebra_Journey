# ML_Linear_Algebra_Journey
Project Goal: SA One-Week Journey into Linear Algebra for ML.

1. What is the core idea of representing documents as vectors?

The core idea of representing documents as vectors is to convert text data into numerical form so that it can be analyzed mathematically. Each document is transformed into a vector in a high-dimensional space, where each dimension often corresponds to a unique word or feature extracted from the text. This allows the semantic content of the document to be captured in a structured way, making it suitable for computation and comparison.

2. Why is this representation useful for tasks like search engines or document comparison?

This representation is useful for tasks like search engines or document comparison because it enables efficient measurement of similarity between documents. By representing text as vectors, algorithms can quickly find documents that are similar to a query or cluster related documents together. This vector-based approach allows machines to “understand” relationships between documents beyond simple keyword matching.

3. What are at least two different mathematical methods for comparing these vectors (e.g., Cosine Similarity, Euclidean Distance), and what is the conceptual difference between them?

There are several mathematical methods to compare these vectors. Cosine Similarity measures the cosine of the angle between two vectors, focusing on the direction rather than the magnitude. This is especially useful when the overall length of documents varies, but the relative distribution of words matters. Euclidean Distance, on the other hand, measures the straight-line distance between vectors in space, taking both magnitude and direction into account. Conceptually, Cosine Similarity is about orientation (how similar the patterns are), whereas Euclidean Distance is about absolute difference in values (how far apart they are in space).