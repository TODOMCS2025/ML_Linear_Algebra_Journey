"""
Tests for the linear algebra utilities module.

These tests demonstrate proper testing practices and provide examples
for adding tests to other modules in the project.
"""

import numpy as np
import pytest
from day6_linalg_utils import (
    calculate_dot_product,
    calculate_l1_norm,
    calculate_l2_norm,
    calculate_angle_between_vectors,
    calculate_vector_projection,
    calculate_euclidean_distance,
    calculate_cosine_similarity,
    calculate_matrix_determinant,
    calculate_matrix_inverse,
    is_orthogonal,
    is_symmetric,
)


class TestVectorOperations:
    """Test vector operations functions."""
    
    def test_dot_product_basic(self):
        """Test basic dot product calculation."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        result = calculate_dot_product(v1, v2)
        expected = 32.0  # 1*4 + 2*5 + 3*6
        assert result == expected
    
    def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors."""
        v1 = [1, 0]
        v2 = [0, 1]
        result = calculate_dot_product(v1, v2)
        assert result == 0.0
    
    def test_dot_product_shape_mismatch(self):
        """Test that dot product raises error for mismatched shapes."""
        v1 = [1, 2]
        v2 = [1, 2, 3]
        with pytest.raises(ValueError, match="same shape"):
            calculate_dot_product(v1, v2)
    
    def test_l1_norm(self):
        """Test L1 norm calculation."""
        vector = [3, -4, 0]
        result = calculate_l1_norm(vector)
        expected = 7.0  # |3| + |-4| + |0|
        assert result == expected
    
    def test_l2_norm(self):
        """Test L2 norm calculation."""
        vector = [3, 4, 0]
        result = calculate_l2_norm(vector)
        expected = 5.0  # sqrt(3^2 + 4^2 + 0^2)
        assert result == expected
    
    def test_angle_between_vectors_orthogonal(self):
        """Test angle between orthogonal vectors."""
        v1 = [1, 0]
        v2 = [0, 1]
        result = calculate_angle_between_vectors(v1, v2)
        expected = np.pi / 2
        assert abs(result - expected) < 1e-10
    
    def test_angle_between_vectors_parallel(self):
        """Test angle between parallel vectors."""
        v1 = [1, 2, 3]
        v2 = [2, 4, 6]  # Same direction, different magnitude
        result = calculate_angle_between_vectors(v1, v2)
        assert abs(result) < 1e-10  # Should be very close to 0
    
    def test_angle_zero_vector_raises_error(self):
        """Test that zero vector raises error in angle calculation."""
        v1 = [0, 0]
        v2 = [1, 0]
        with pytest.raises(ValueError, match="zero-magnitude"):
            calculate_angle_between_vectors(v1, v2)
    
    def test_vector_projection(self):
        """Test vector projection calculation."""
        v1 = [2, 1]
        v2 = [1, 0]  # Project onto x-axis
        result = calculate_vector_projection(v1, v2)
        expected = np.array([2.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_vector_projection_zero_target(self):
        """Test that projection onto zero vector raises error."""
        v1 = [1, 2]
        v2 = [0, 0]
        with pytest.raises(ValueError, match="zero vector"):
            calculate_vector_projection(v1, v2)


class TestDistanceAndSimilarity:
    """Test distance and similarity functions."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        v1 = [0, 0]
        v2 = [3, 4]
        result = calculate_euclidean_distance(v1, v2)
        expected = 5.0
        assert result == expected
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        result = calculate_cosine_similarity(v1, v2)
        assert abs(result - 1.0) < 1e-10
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        v1 = [1, 0]
        v2 = [0, 1]
        result = calculate_cosine_similarity(v1, v2)
        assert abs(result) < 1e-10


class TestMatrixOperations:
    """Test matrix operations functions."""
    
    def test_matrix_determinant_2x2(self):
        """Test determinant of 2x2 matrix."""
        matrix = [[2, 1], [1, 3]]
        result = calculate_matrix_determinant(matrix)
        expected = 5.0  # 2*3 - 1*1
        assert abs(result - expected) < 1e-10
    
    def test_matrix_determinant_non_square(self):
        """Test that non-square matrix raises error."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="square"):
            calculate_matrix_determinant(matrix)
    
    def test_matrix_inverse_invertible(self):
        """Test inverse of invertible matrix."""
        matrix = [[2, 1], [1, 3]]
        result = calculate_matrix_inverse(matrix)
        assert result is not None
        
        # Verify A * A^-1 = I
        matrix_np = np.array(matrix)
        product = matrix_np @ result
        identity = np.eye(2)
        np.testing.assert_array_almost_equal(product, identity)
    
    def test_matrix_inverse_singular(self):
        """Test that singular matrix returns None."""
        matrix = [[1, 2], [2, 4]]  # Singular matrix
        result = calculate_matrix_inverse(matrix)
        assert result is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_is_orthogonal_true(self):
        """Test orthogonal vectors detection."""
        v1 = [1, 0]
        v2 = [0, 1]
        assert is_orthogonal(v1, v2) is True
    
    def test_is_orthogonal_false(self):
        """Test non-orthogonal vectors detection."""
        v1 = [1, 1]
        v2 = [1, 0]
        assert is_orthogonal(v1, v2) is False
    
    def test_is_symmetric_true(self):
        """Test symmetric matrix detection."""
        matrix = [[1, 2], [2, 3]]
        assert is_symmetric(matrix) is True
    
    def test_is_symmetric_false(self):
        """Test non-symmetric matrix detection."""
        matrix = [[1, 2], [3, 4]]
        assert is_symmetric(matrix) is False


# Fixtures for common test data
@pytest.fixture
def sample_vectors():
    """Provide sample vectors for testing."""
    return {
        'v1': np.array([1, 2, 3]),
        'v2': np.array([4, 5, 6]),
        'zero': np.array([0, 0, 0]),
        'unit_x': np.array([1, 0, 0]),
        'unit_y': np.array([0, 1, 0]),
    }


@pytest.fixture
def sample_matrices():
    """Provide sample matrices for testing."""
    return {
        'identity_2x2': np.eye(2),
        'symmetric': np.array([[1, 2], [2, 3]]),
        'invertible': np.array([[2, 1], [1, 3]]),
        'singular': np.array([[1, 2], [2, 4]]),
    }


if __name__ == "__main__":
    pytest.main([__file__])
