### Day 6: Comprehensive Linear Algebra Utilities - All-in-One Module
### A collection of essential linear algebra functions
### Vector operations, matrix operations, decompositions, and more
### Date: 2025-08-14

import numpy as np # Import dependencies at the top

def calculate_dot_product(v1, v2):
    return np.dot(v1, v2)

def calculate_l1_norm(vector):
    return np.linalg.norm(vector, 1)

def calculate_l2_norm(vector):
    return np.linalg.norm(vector)

def calculate_angle_between_vectors(v1, v2):
    cosangle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosangle, -1.0, 1.0))  # Clip to avoid numerical issues
    return angle

def calculate_vector_projection(v1, v2):
    proj = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
    return proj

# === DISTANCE AND SIMILARITY MEASURES ===
def calculate_euclidean_distance(v1, v2):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(v1 - v2)

def calculate_manhattan_distance(v1, v2):
    """Calculate Manhattan (L1) distance between two vectors"""
    return np.sum(np.abs(v1 - v2))

def calculate_cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors (0 to 1)"""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_product / norms if norms != 0 else 0

def calculate_cosine_distance(v1, v2):
    """Calculate cosine distance (1 - cosine_similarity)"""
    return 1 - calculate_cosine_similarity(v1, v2)

# === MATRIX OPERATIONS ===
def calculate_matrix_determinant(matrix):
    """Calculate the determinant of a matrix"""
    return np.linalg.det(matrix)

def calculate_matrix_inverse(matrix):
    """Calculate the inverse of a matrix (if invertible)"""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None  # Matrix is not invertible

def calculate_matrix_rank(matrix):
    """Calculate the rank of a matrix"""
    return np.linalg.matrix_rank(matrix)

def calculate_matrix_trace(matrix):
    """Calculate the trace (sum of diagonal elements) of a matrix"""
    return np.trace(matrix)

def transpose_matrix(matrix):
    """Calculate the transpose of a matrix"""
    return matrix.T

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

# === UTILITY FUNCTIONS ===
def is_orthogonal(v1, v2, tolerance=1e-10):
    """Check if two vectors are orthogonal"""
    return abs(np.dot(v1, v2)) < tolerance

def is_symmetric(matrix, tolerance=1e-10):
    """Check if a matrix is symmetric"""
    return np.allclose(matrix, matrix.T, atol=tolerance)

def is_positive_definite(matrix):
    """Check if a matrix is positive definite"""
    try:
        np.linalg.cholesky(matrix)
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

# === DEMONSTRATION AND TESTING ===
if __name__ == "__main__":
    print("=== LINEAR ALGEBRA UTILITIES DEMO ===\n")
    
    # Test vectors
    v1 = np.array([3, 4, 0])
    v2 = np.array([1, 2, 2])
    
    print("--- VECTOR OPERATIONS ---")
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Dot product: {calculate_dot_product(v1, v2)}")
    print(f"L1 norm (v1): {calculate_l1_norm(v1)}")
    print(f"L2 norm (v1): {calculate_l2_norm(v1)}")
    print(f"Angle between vectors (radians): {calculate_angle_between_vectors(v1, v2):.3f}")
    print(f"Angle between vectors (degrees): {np.degrees(calculate_angle_between_vectors(v1, v2)):.3f}")
    print(f"Euclidean distance: {calculate_euclidean_distance(v1, v2):.3f}")
    print(f"Cosine similarity: {calculate_cosine_similarity(v1, v2):.3f}")
    print(f"Unit vector (v1): {unit_vector(v1)}")
    
    if len(v1) == 3 and len(v2) == 3:
        print(f"Cross product: {cross_product(v1, v2)}")
    
    print(f"\n--- MATRIX OPERATIONS ---")
    # Test matrix
    matrix = np.array([[2, 1], [1, 3]])
    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {calculate_matrix_determinant(matrix):.3f}")
    print(f"Trace: {calculate_matrix_trace(matrix)}")
    print(f"Rank: {calculate_matrix_rank(matrix)}")
    
    inv_matrix = calculate_matrix_inverse(matrix)
    if inv_matrix is not None:
        print(f"Inverse:\n{inv_matrix}")
        print(f"Verification (A × A⁻¹):\n{np.round(matrix @ inv_matrix, 10)}")
    
    print(f"Is symmetric: {is_symmetric(matrix)}")
    print(f"Is positive definite: {is_positive_definite(matrix)}")
    print(f"Condition number: {condition_number(matrix):.3f}")
    
    print(f"\n--- EIGENVALUES AND EIGENVECTORS ---")
    eigenvals, eigenvecs = calculate_eigenvalues_eigenvectors(matrix)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    print(f"\n--- MATRIX DECOMPOSITIONS ---")
    U, s, Vt = singular_value_decomposition(matrix)
    print(f"SVD - Singular values: {s}")
    
    Q, R = qr_decomposition(matrix)
    print(f"QR Decomposition verification (Q × R):\n{np.round(Q @ R, 10)}")
    
    print(f"\n--- LINEAR SYSTEM SOLVING ---")
    A = np.array([[2, 1], [1, 3]])
    b = np.array([3, 4])
    solution = solve_linear_system(A, b)
    if solution is not None:
        print(f"System Ax = b where A =\n{A}")
        print(f"b = {b}")
        print(f"Solution x = {solution}")
        print(f"Verification Ax = {A @ solution}")
    
    print(f"\n--- UTILITY MATRICES ---")
    print(f"2D Rotation matrix (45°):\n{create_rotation_matrix_2d(np.pi/4)}")
    print(f"3x3 Identity matrix:\n{create_identity_matrix(3)}")
    
    print(f"\n--- ORTHOGONALITY TESTS ---")
    orth_v1 = np.array([1, 0])
    orth_v2 = np.array([0, 1])
    print(f"Vectors {orth_v1} and {orth_v2} are orthogonal: {is_orthogonal(orth_v1, orth_v2)}")
    
    print(f"\n=== ALL FUNCTIONS IMPLEMENTED ===")
    functions = [func for func in dir() if callable(globals()[func]) and not func.startswith('_') and func != 'np']
    print(f"Total functions available: {len(functions)}")
    for i, func in enumerate(sorted(functions), 1):
        print(f"{i:2d}. {func}")
    
    print(f"\nLinear algebra utilities ready for use! 🚀")