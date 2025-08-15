# ML Linear Algebra Journey 🧮

A comprehensive collection of linear algebra utilities and learning exercises designed for machine learning applications. This repository provides both educational content and production-ready utilities for vector and matrix operations.

## 📚 Overview

This project is structured as a 6-day journey through linear algebra fundamentals, progressing from basic vector operations to advanced matrix decompositions and utility functions.

### Learning Path

- **Day 1**: [Vector Operations Basics](day1_vectorops_practice.py) - Foundational vector operations with lists and NumPy
- **Day 2**: [Core Vector Operations](day2_vectorops_practice.py) - NumPy arrays and advanced vector operations  
- **Day 3**: [Matrix Operations](day3_matrixops_practice.py) - Matrix arithmetic and transformations
- **Day 4**: [Advanced Linear Algebra](day4_advanced_linalg.py) - Eigenvalues, decompositions, and advanced concepts
- **Day 5**: [Similarity Analysis](day5_similarity_analysis.py) - Distance metrics and similarity measures
- **Day 6**: [Comprehensive Utilities](day6_linalg_utils.py) - Production-ready linear algebra library

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/TODOMCS2025/ML_Linear_Algebra_Journey.git
cd ML_Linear_Algebra_Journey
```

2. Install dependencies:
```bash
pip install numpy
```

3. For development (optional):
```bash
pip install -e ".[dev]"
```

### Basic Usage

```python
from day6_linalg_utils import (
    calculate_dot_product,
    calculate_cosine_similarity,
    calculate_matrix_inverse,
    solve_linear_system
)

# Vector operations
v1 = [1, 2, 3]
v2 = [4, 5, 6]
dot_product = calculate_dot_product(v1, v2)
similarity = calculate_cosine_similarity(v1, v2)

# Matrix operations
matrix = [[2, 1], [1, 3]]
inverse = calculate_matrix_inverse(matrix)

# Linear systems
A = [[2, 1], [1, 3]]
b = [3, 4]
solution = solve_linear_system(A, b)
```

### Run Demo

```bash
python day6_linalg_utils.py
```

## 📖 Features

### Vector Operations
- Dot product, cross product
- L1, L2 norms and normalization
- Angle calculations and projections
- Unit vector computation

### Distance & Similarity Metrics
- Euclidean and Manhattan distance
- Cosine similarity and distance
- Robust handling of edge cases

### Matrix Operations
- Determinant, inverse, rank, trace
- Transpose and matrix powers
- Condition number analysis
- Comprehensive error handling

### Matrix Decompositions
- Singular Value Decomposition (SVD)
- QR decomposition
- Cholesky decomposition
- Eigenvalue/eigenvector computation

### Linear Systems
- Direct solving (Ax = b)
- Least squares solutions
- Robust error handling for singular systems

### Advanced Features
- Gram-Schmidt orthogonalization
- Matrix property checks (symmetric, positive definite)
- Rotation matrix generation
- Utility matrix creation

## 🛠️ Development

### Code Quality

This project uses modern Python development practices:

- **Type Hints**: Full type annotations for better IDE support
- **Docstrings**: Comprehensive documentation with examples
- **Input Validation**: Robust error handling and input checking
- **Testing**: Unit tests with pytest (run `pytest`)
- **Formatting**: Black for code formatting (`black .`)
- **Linting**: Ruff for fast linting (`ruff check .`)
- **Type Checking**: MyPy for static type analysis (`mypy .`)

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=.

# Run formatting
black .

# Run linting
ruff check .

# Run type checking
mypy .
```

### Project Structure

```
ML_Linear_Algebra_Journey/
├── day1_vectorops_practice.py      # Basic vector operations
├── day2_vectorops_practice.py      # NumPy vector operations  
├── day3_matrixops_practice.py      # Matrix operations
├── day4_advanced_linalg.py         # Advanced concepts
├── day5_similarity_analysis.py     # Similarity measures
├── day6_linalg_utils.py            # Complete utilities library
├── test_day6_linalg_utils.py       # Unit tests
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```

## 📊 Examples

### Vector Analysis
```python
import numpy as np
from day6_linalg_utils import *

# Create vectors
v1 = np.array([3, 4, 0])
v2 = np.array([1, 2, 2])

# Calculate various metrics
print(f"Dot product: {calculate_dot_product(v1, v2)}")
print(f"Angle (degrees): {np.degrees(calculate_angle_between_vectors(v1, v2)):.1f}°")
print(f"Cosine similarity: {calculate_cosine_similarity(v1, v2):.3f}")
print(f"Euclidean distance: {calculate_euclidean_distance(v1, v2):.3f}")
```

### Matrix Analysis
```python
# Create matrix
matrix = np.array([[2, 1], [1, 3]])

# Analyze properties
print(f"Determinant: {calculate_matrix_determinant(matrix)}")
print(f"Is symmetric: {is_symmetric(matrix)}")
print(f"Condition number: {condition_number(matrix):.2f}")

# Decompose
eigenvals, eigenvecs = calculate_eigenvalues_eigenvectors(matrix)
U, s, Vt = singular_value_decomposition(matrix)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code style guidelines
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style Guidelines

- Use type hints for all function parameters and return values
- Write comprehensive docstrings with examples
- Add input validation and appropriate error messages
- Include unit tests for new functions
- Follow PEP 8 style guidelines (enforced by Black)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🎯 Future Enhancements

- [ ] Sparse matrix operations
- [ ] GPU acceleration with CuPy
- [ ] Advanced decompositions (LU, Schur)
- [ ] Performance benchmarking suite
- [ ] Interactive Jupyter notebooks
- [ ] Visualization utilities
- [ ] Integration with scikit-learn

## 📚 Learning Resources

- [Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) - Stanford CS229
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - Comprehensive matrix reference
- [NumPy Documentation](https://numpy.org/doc/stable/) - Official NumPy documentation

---

*Happy learning! 🚀*
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