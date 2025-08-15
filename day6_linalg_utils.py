"""
Day 6: Comprehensive Linear Algebra Utilities - All-in-One Module

A collection of essential linear algebra functions including:
- Vector operations (dot product, norms, projections)
- Matrix operations (determinant, inverse, decompositions)
- Distance and similarity measures
- Linear system solving
- Utility functions

Author: ML Linear Algebra Journey
Date: 2025-08-14
"""

from typing import Optional, Tuple, Union, Literal
import numpy as np
import numpy.typing as npt

# Type aliases for better readability
Vector = npt.NDArray[np.floating]
Matrix = npt.NDArray[np.floating]
ArrayLike = Union[list, tuple, npt.NDArray]

# Public API
__all__ = [
    # Vector operations
    'calculate_dot_product', 'calculate_l1_norm', 'calculate_l2_norm',
    'calculate_angle_between_vectors', 'calculate_vector_projection',
    'normalize_vector', 'cross_product', 'vector_magnitude', 'unit_vector',
    
    # Distance and similarity
    'calculate_euclidean_distance', 'calculate_manhattan_distance',
    'calculate_cosine_similarity', 'calculate_cosine_distance',
    
    # Matrix operations
    'calculate_matrix_determinant', 'calculate_matrix_inverse',
    'calculate_matrix_rank', 'calculate_matrix_trace', 'transpose_matrix',
    
    # Eigenvalues and decompositions
    'calculate_eigenvalues_eigenvectors', 'singular_value_decomposition',
    'qr_decomposition', 'cholesky_decomposition',
    
    # Advanced operations
    'gram_schmidt_process', 'condition_number', 'matrix_power',
    
    # Linear systems
    'solve_linear_system', 'least_squares_solution',
    
    # Utility functions
    'is_orthogonal', 'is_symmetric', 'is_positive_definite',
    'create_rotation_matrix_2d', 'create_identity_matrix',
    'create_zero_matrix', 'create_ones_matrix'
]

# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def calculate_dot_product(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Calculate the dot product of two vectors.
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        
    Returns:
        Dot product as a scalar value
        
    Raises:
        ValueError: If vectors have different lengths
        
    Example:
        >>> calculate_dot_product([1, 2, 3], [4, 5, 6])
        32.0
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    return float(np.dot(v1_arr, v2_arr))


def calculate_l1_norm(vector: ArrayLike) -> float:
    """
    Calculate the L1 norm (Manhattan norm) of a vector.
    
    Args:
        vector: Input vector (array-like)
        
    Returns:
        L1 norm as a scalar value
        
    Example:
        >>> calculate_l1_norm([3, 4, 0])
        7.0
    """
    vector_arr = np.asarray(vector, dtype=float)
    return float(np.linalg.norm(vector_arr, 1))


def calculate_l2_norm(vector: ArrayLike) -> float:
    """
    Calculate the L2 norm (Euclidean norm) of a vector.
    
    Args:
        vector: Input vector (array-like)
        
    Returns:
        L2 norm as a scalar value
        
    Example:
        >>> calculate_l2_norm([3, 4, 0])
        5.0
    """
    vector_arr = np.asarray(vector, dtype=float)
    return float(np.linalg.norm(vector_arr))


def calculate_angle_between_vectors(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Calculate the angle between two vectors in radians.
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        
    Returns:
        Angle in radians (0 to π)
        
    Raises:
        ValueError: If either vector has zero magnitude
        
    Example:
        >>> import numpy as np
        >>> calculate_angle_between_vectors([1, 0], [0, 1])
        1.5707963267948966  # π/2
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    norm1 = np.linalg.norm(v1_arr)
    norm2 = np.linalg.norm(v2_arr)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot calculate angle with zero-magnitude vector")
    
    cos_angle = np.dot(v1_arr, v2_arr) / (norm1 * norm2)
    # Clip to avoid numerical issues with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def calculate_vector_projection(v1: ArrayLike, v2: ArrayLike) -> Vector:
    """
    Calculate the projection of vector v1 onto vector v2.
    
    Args:
        v1: Vector to be projected (array-like)
        v2: Vector to project onto (array-like)
        
    Returns:
        Projection vector as numpy array
        
    Raises:
        ValueError: If v2 has zero magnitude
        
    Example:
        >>> calculate_vector_projection([1, 2], [1, 0])
        array([1., 0.])
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    v2_norm_squared = np.dot(v2_arr, v2_arr)
    if v2_norm_squared == 0:
        raise ValueError("Cannot project onto zero vector")
    
    projection_scalar = np.dot(v1_arr, v2_arr) / v2_norm_squared
    return projection_scalar * v2_arr

# ============================================================================
# DISTANCE AND SIMILARITY MEASURES
# ============================================================================

def calculate_euclidean_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        
    Returns:
        Euclidean distance as a scalar
        
    Example:
        >>> calculate_euclidean_distance([0, 0], [3, 4])
        5.0
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    return float(np.linalg.norm(v1_arr - v2_arr))


def calculate_manhattan_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Calculate Manhattan (L1) distance between two vectors.
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        
    Returns:
        Manhattan distance as a scalar
        
    Example:
        >>> calculate_manhattan_distance([0, 0], [3, 4])
        7.0
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    return float(np.sum(np.abs(v1_arr - v2_arr)))


def calculate_cosine_similarity(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        
    Returns:
        Cosine similarity (-1 to 1)
        
    Note:
        Returns 0 if either vector has zero magnitude
        
    Example:
        >>> calculate_cosine_similarity([1, 0], [0, 1])
        0.0
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    dot_product = np.dot(v1_arr, v2_arr)
    norms = np.linalg.norm(v1_arr) * np.linalg.norm(v2_arr)
    
    return float(dot_product / norms if norms != 0 else 0)


def calculate_cosine_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Calculate cosine distance (1 - cosine_similarity).
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        
    Returns:
        Cosine distance (0 to 2)
        
    Example:
        >>> calculate_cosine_distance([1, 0], [0, 1])
        1.0
    """
    return 1.0 - calculate_cosine_similarity(v1, v2)

# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def calculate_matrix_determinant(matrix: ArrayLike) -> float:
    """
    Calculate the determinant of a square matrix.
    
    Args:
        matrix: Square matrix (array-like)
        
    Returns:
        Determinant as a scalar
        
    Raises:
        ValueError: If matrix is not square
        
    Example:
        >>> calculate_matrix_determinant([[2, 1], [1, 3]])
        5.0
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix_arr.shape}")
    
    return float(np.linalg.det(matrix_arr))


def calculate_matrix_inverse(matrix: ArrayLike) -> Optional[Matrix]:
    """
    Calculate the inverse of a square matrix.
    
    Args:
        matrix: Square matrix (array-like)
        
    Returns:
        Inverse matrix if invertible, None otherwise
        
    Raises:
        ValueError: If matrix is not square
        
    Example:
        >>> inv = calculate_matrix_inverse([[2, 1], [1, 3]])
        >>> inv is not None
        True
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix_arr.shape}")
    
    try:
        return np.linalg.inv(matrix_arr)
    except np.linalg.LinAlgError:
        return None


def calculate_matrix_rank(matrix: ArrayLike) -> int:
    """
    Calculate the rank of a matrix.
    
    Args:
        matrix: Input matrix (array-like)
        
    Returns:
        Rank of the matrix
        
    Example:
        >>> calculate_matrix_rank([[1, 2], [2, 4]])
        1
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    return int(np.linalg.matrix_rank(matrix_arr))


def calculate_matrix_trace(matrix: ArrayLike) -> float:
    """
    Calculate the trace (sum of diagonal elements) of a square matrix.
    
    Args:
        matrix: Square matrix (array-like)
        
    Returns:
        Trace as a scalar
        
    Raises:
        ValueError: If matrix is not square
        
    Example:
        >>> calculate_matrix_trace([[2, 1], [1, 3]])
        5.0
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix_arr.shape}")
    
    return float(np.trace(matrix_arr))


def transpose_matrix(matrix: ArrayLike) -> Matrix:
    """
    Calculate the transpose of a matrix.
    
    Args:
        matrix: Input matrix (array-like)
        
    Returns:
        Transposed matrix
        
    Example:
        >>> transpose_matrix([[1, 2], [3, 4]])
        array([[1., 3.],
               [2., 4.]])
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    return matrix_arr.T

# === EIGENVALUES AND EIGENVECTORS ===
def calculate_eigenvalues_eigenvectors(matrix):
    """Calculate eigenvalues and eigenvectors of a matrix"""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

# === MATRIX DECOMPOSITIONS ===
def singular_value_decomposition(matrix):
    """Perform Singular Value Decomposition (SVD)"""
    U, s, Vt = np.linalg.svd(matrix)
    return U, s, Vt

def qr_decomposition(matrix):
    """Perform QR decomposition"""
    Q, R = np.linalg.qr(matrix)
    return Q, R

def cholesky_decomposition(matrix):
    """Perform Cholesky decomposition (for positive definite matrices)"""
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return None  # Matrix is not positive definite

# === VECTOR OPERATIONS ===
def normalize_vector(vector, norm='l2'):
    """Normalize a vector using L1 or L2 norm"""
    if norm == 'l1':
        norm_val = np.sum(np.abs(vector))
    elif norm == 'l2':
        norm_val = np.linalg.norm(vector)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")
    
    return vector / norm_val if norm_val != 0 else vector

def cross_product(v1, v2):
    """Calculate cross product of two 3D vectors"""
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product only defined for 3D vectors")
    return np.cross(v1, v2)

def vector_magnitude(vector):
    """Calculate the magnitude (L2 norm) of a vector"""
    return np.linalg.norm(vector)

def unit_vector(vector):
    """Calculate the unit vector (normalized to length 1)"""
    return normalize_vector(vector, norm='l2')

# === ADVANCED OPERATIONS ===
def gram_schmidt_process(vectors):
    """Apply Gram-Schmidt process to orthogonalize vectors"""
    orthogonal_vectors = []
    for v in vectors:
        # Subtract projections onto previous orthogonal vectors
        for u in orthogonal_vectors:
            projection = calculate_vector_projection(v, u)
            v = v - projection
        
        # Normalize the resulting vector
        if np.linalg.norm(v) > 1e-10:  # Avoid division by zero
            orthogonal_vectors.append(unit_vector(v))
    
    return np.array(orthogonal_vectors)

def condition_number(matrix):
    """Calculate the condition number of a matrix"""
    return np.linalg.cond(matrix)

def matrix_power(matrix, power):
    """Calculate matrix raised to a power"""
    return np.linalg.matrix_power(matrix, power)

# === LINEAR SYSTEM SOLVING ===
def solve_linear_system(A, b):
    """Solve the linear system Ax = b"""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # System has no unique solution

def least_squares_solution(A, b):
    """Find least squares solution to Ax = b"""
    return np.linalg.lstsq(A, b, rcond=None)[0]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_orthogonal(v1: ArrayLike, v2: ArrayLike, tolerance: float = 1e-10) -> bool:
    """
    Check if two vectors are orthogonal.
    
    Args:
        v1: First vector (array-like)
        v2: Second vector (array-like)
        tolerance: Numerical tolerance for orthogonality check
        
    Returns:
        True if vectors are orthogonal, False otherwise
        
    Example:
        >>> is_orthogonal([1, 0], [0, 1])
        True
    """
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    
    if v1_arr.shape != v2_arr.shape:
        raise ValueError(f"Vectors must have same shape: {v1_arr.shape} vs {v2_arr.shape}")
    
    dot_product = np.dot(v1_arr, v2_arr)
    return bool(abs(dot_product) < tolerance)


def is_symmetric(matrix: ArrayLike, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is symmetric.
    
    Args:
        matrix: Input matrix (array-like)
        tolerance: Numerical tolerance for symmetry check
        
    Returns:
        True if matrix is symmetric, False otherwise
        
    Raises:
        ValueError: If matrix is not square
        
    Example:
        >>> is_symmetric([[1, 2], [2, 3]])
        True
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix_arr.shape}")
    
    return bool(np.allclose(matrix_arr, matrix_arr.T, atol=tolerance))


def is_positive_definite(matrix: ArrayLike) -> bool:
    """
    Check if a matrix is positive definite.
    
    Args:
        matrix: Square matrix (array-like)
        
    Returns:
        True if matrix is positive definite, False otherwise
        
    Raises:
        ValueError: If matrix is not square
        
    Example:
        >>> is_positive_definite([[2, 1], [1, 2]])
        True
    """
    matrix_arr = np.asarray(matrix, dtype=float)
    
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix_arr.shape}")
    
    try:
        np.linalg.cholesky(matrix_arr)
        return True
    except np.linalg.LinAlgError:
        return False

def create_rotation_matrix_2d(angle_radians):
    """Create 2D rotation matrix"""
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

def create_identity_matrix(size):
    """Create identity matrix of given size"""
    return np.eye(size)

def create_zero_matrix(rows, cols):
    """Create zero matrix of given dimensions"""
    return np.zeros((rows, cols))

def create_ones_matrix(rows, cols):
    """Create matrix filled with ones"""
    return np.ones((rows, cols))

# ============================================================================
# DEMONSTRATION AND EXAMPLES
# ============================================================================

def demo() -> None:
    """
    Demonstrate the functionality of the linear algebra utilities.
    
    This function provides examples of how to use the various functions
    in this module and showcases their capabilities.
    """
    print("=" * 60)
    print("LINEAR ALGEBRA UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    _demo_vector_operations()
    _demo_matrix_operations()
    _demo_decompositions()
    _demo_linear_systems()
    _demo_utility_functions()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


def _demo_vector_operations() -> None:
    """Demonstrate vector operations."""
    print("\n" + "-" * 40)
    print("VECTOR OPERATIONS")
    print("-" * 40)
    
    v1 = np.array([3, 4, 0])
    v2 = np.array([1, 2, 2])
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Dot product: {calculate_dot_product(v1, v2)}")
    print(f"L1 norm (v1): {calculate_l1_norm(v1):.3f}")
    print(f"L2 norm (v1): {calculate_l2_norm(v1):.3f}")
    print(f"Angle (radians): {calculate_angle_between_vectors(v1, v2):.3f}")
    print(f"Angle (degrees): {np.degrees(calculate_angle_between_vectors(v1, v2)):.1f}°")
    print(f"Euclidean distance: {calculate_euclidean_distance(v1, v2):.3f}")
    print(f"Cosine similarity: {calculate_cosine_similarity(v1, v2):.3f}")
    print(f"Unit vector (v1): {unit_vector(v1)}")
    
    if len(v1) == 3 and len(v2) == 3:
        print(f"Cross product: {cross_product(v1, v2)}")


def _demo_matrix_operations() -> None:
    """Demonstrate matrix operations."""
    print("\n" + "-" * 40)
    print("MATRIX OPERATIONS")
    print("-" * 40)
    
    matrix = np.array([[2, 1], [1, 3]], dtype=float)
    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {calculate_matrix_determinant(matrix):.3f}")
    print(f"Trace: {calculate_matrix_trace(matrix)}")
    print(f"Rank: {calculate_matrix_rank(matrix)}")
    
    inv_matrix = calculate_matrix_inverse(matrix)
    if inv_matrix is not None:
        print(f"Inverse:\n{inv_matrix}")
        verification = matrix @ inv_matrix
        print(f"Verification (A × A⁻¹):\n{np.round(verification, 10)}")
    
    print(f"Is symmetric: {is_symmetric(matrix)}")
    print(f"Is positive definite: {is_positive_definite(matrix)}")
    print(f"Condition number: {condition_number(matrix):.3f}")


def _demo_decompositions() -> None:
    """Demonstrate matrix decompositions."""
    print("\n" + "-" * 40)
    print("MATRIX DECOMPOSITIONS")
    print("-" * 40)
    
    matrix = np.array([[2, 1], [1, 3]], dtype=float)
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = calculate_eigenvalues_eigenvectors(matrix)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # SVD
    U, s, Vt = singular_value_decomposition(matrix)
    print(f"SVD - Singular values: {s}")
    
    # QR decomposition
    Q, R = qr_decomposition(matrix)
    verification = Q @ R
    print(f"QR verification (Q × R):\n{np.round(verification, 10)}")


def _demo_linear_systems() -> None:
    """Demonstrate linear system solving."""
    print("\n" + "-" * 40)
    print("LINEAR SYSTEM SOLVING")
    print("-" * 40)
    
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([3, 4], dtype=float)
    
    solution = solve_linear_system(A, b)
    if solution is not None:
        print(f"System Ax = b where A =\n{A}")
        print(f"b = {b}")
        print(f"Solution x = {solution}")
        print(f"Verification Ax = {A @ solution}")


def _demo_utility_functions() -> None:
    """Demonstrate utility functions."""
    print("\n" + "-" * 40)
    print("UTILITY FUNCTIONS")
    print("-" * 40)
    
    # Rotation matrix
    rotation_45 = create_rotation_matrix_2d(np.pi/4)
    print(f"2D Rotation matrix (45°):\n{np.round(rotation_45, 3)}")
    
    # Identity matrix
    identity_3x3 = create_identity_matrix(3)
    print(f"3×3 Identity matrix:\n{identity_3x3}")
    
    # Orthogonality test
    orth_v1 = np.array([1, 0])
    orth_v2 = np.array([0, 1])
    print(f"Vectors {orth_v1} and {orth_v2} are orthogonal: {is_orthogonal(orth_v1, orth_v2)}")


if __name__ == "__main__":
    demo()