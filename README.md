# ML_Linear_Algebra_Journey
Project Goal: SA One-Week Journey into Linear Algebra for ML.

Week 1 Retrospective

Study concepts in linear algebra, such as the definition of vectors and basic operations: addition, subtraction, multiplication, scalar multiplication, and vector magnitude. This includes their corresponding implementation in Python using the NumPy library, which provides functions for various mathematical operations. We also covered how to calculate the L1 norm, L2 norm, and squared L2 norm.

On the other hand, review topics related to matrices and their operations—similar to those mentioned for vectors—and their implementation with the NumPy library.

Additionally, study the inverse and determinants of 2×2 and 3×3 matrices, as well as how to solve linear equations using matrices. It was interesting to remember and refresh high school concepts and then apply advanced techniques using computational software.

Another important topic this week was similarity analysis and the study of the Vector Space Model. For this exercise, we defined an array of documents. It was necessary to create a vocabulary containing all the words present in every document. The next step was to define vectors for those documents, indicating the positions and frequency of each word in every document. We then counted the number of word matches and the number of unique words. Finally, we explained the norm and similarity of these word vectors by comparing them pairwise.

For all these main topics, I reviewed YouTube videos from different authors that explained both the theory and Python implementations, as well as shared fresh information about their applications.

    a. Foundations of Linear Algebra  (day1_vectorops_practice.py)
    b. Core Vector Operations with Numpy (day2_vectorops_practice.py)
    c. Matrices (day3_matrixops_practice.py)
    d. Inverses and Determinants (day3_advanced_linalg.py)
    e. Applied Analysis -Quantifying Semantic Similarity (day5_similarity_analysis.py)
    f. Building a Verified Vector anlysis Toolkit (day6_linalg_utils.py)



1. What is the core idea of representing documents as vectors?

The core idea of representing documents as vectors is to convert text data into numerical form so that it can be analyzed mathematically. Each document is transformed into a vector in a high-dimensional space, where each dimension often corresponds to a unique word or feature extracted from the text. This allows the semantic content of the document to be captured in a structured way, making it suitable for computation and comparison.

2. Why is this representation useful for tasks like search engines or document comparison?

This representation is useful for tasks like search engines or document comparison because it enables efficient measurement of similarity between documents. By representing text as vectors, algorithms can quickly find documents that are similar to a query or cluster related documents together. This vector-based approach allows machines to “understand” relationships between documents beyond simple keyword matching.

3. What are at least two different mathematical methods for comparing these vectors (e.g., Cosine Similarity, Euclidean Distance), and what is the conceptual difference between them?

There are several mathematical methods to compare these vectors. Cosine Similarity measures the cosine of the angle between two vectors, focusing on the direction rather than the magnitude. This is especially useful when the overall length of documents varies, but the relative distribution of words matters. Euclidean Distance, on the other hand, measures the straight-line distance between vectors in space, taking both magnitude and direction into account. Conceptually, Cosine Similarity is about orientation (how similar the patterns are), whereas Euclidean Distance is about absolute difference in values (how far apart they are in space).