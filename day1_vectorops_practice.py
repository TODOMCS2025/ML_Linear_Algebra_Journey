"""
Day 1: Foundations of Linear Algebra - Vector Operations Practice

This module introduces basic vector operations using both Python lists
and NumPy arrays. It covers:
- Element-wise operations
- Scalar multiplication  
- Dot product calculations
- Vector magnitude computation

Learning objectives:
- Understand vector representation in Python
- Compare list-based vs NumPy array operations
- Practice fundamental vector arithmetic

Date: 2025-08-06
"""

from typing import List, Union
import numpy as np
import numpy.typing as npt

# Type aliases for better readability
Vector = Union[List[float], npt.NDArray[np.floating]]


def demonstrate_basic_python() -> None:
    """Demonstrate basic Python concepts before diving into linear algebra."""
    print("=" * 50)
    print("BASIC PYTHON DEMONSTRATION")
    print("=" * 50)
    
    # Basic variable assignment
    x = 1
    list_names = ["Alice", "Bob", "Charlie"]
    
    print(f"Variable x: {x}")
    print(f"Names list: {list_names}")
    print()
    
    # Loop through names
    for name in list_names:
        print(f"Hello {name}, welcome to the Python course!")
    print()


def vector_add_manual(a: List[float], b: List[float]) -> List[float]:
    """
    Add two vectors element by element using manual indexing.
    
    Args:
        a: First vector as a list
        b: Second vector as a list
        
    Returns:
        Sum of vectors as a list
        
    Raises:
        ValueError: If vectors have different lengths
        
    Example:
        >>> vector_add_manual([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length: {len(a)} vs {len(b)}")
    
    return [a[i] + b[i] for i in range(len(a))]


def vector_add_zip(a: List[float], b: List[float]) -> List[float]:
    """
    Add two vectors using zip (more Pythonic approach).
    
    Args:
        a: First vector as a list
        b: Second vector as a list
        
    Returns:
        Sum of vectors as a list
        
    Example:
        >>> vector_add_zip([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
    """
    return [x + y for x, y in zip(a, b)]


def demonstrate_list_operations() -> None:
    """Demonstrate vector operations using Python lists."""
    print("=" * 50)
    print("VECTOR OPERATIONS WITH PYTHON LISTS")
    print("=" * 50)
    
    # Define test vectors
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print()
    
    # Vector addition examples
    result_manual = vector_add_manual(v1, v2)
    result_zip = vector_add_zip(v1, v2)
    
    print("Vector Addition Results:")
    print(f"Manual method:    {result_manual}")
    print(f"Zip method:       {result_zip}")
    print()
    
    # Mathematical explanation
    print("Mathematical Explanation:")
    print("v1 + v2 = [1, 2, 3] + [4, 5, 6]")
    print("        = [1+4, 2+5, 3+6]")
    print("        = [5, 7, 9]")
    print()


def demonstrate_numpy_operations() -> None:
    """Demonstrate vector operations using NumPy arrays."""
    print("=" * 50)
    print("VECTOR OPERATIONS WITH NUMPY ARRAYS")
    print("=" * 50)
    
    # Create NumPy arrays
    np_v1 = np.array([1, 2, 3], dtype=float)
    np_v2 = np.array([4, 5, 6], dtype=float)
    
    print(f"NumPy Vector 1: {np_v1}")
    print(f"NumPy Vector 2: {np_v2}")
    print()
    
    # Basic operations (much simpler with NumPy!)
    print("Basic Operations:")
    print(f"Addition:             {np_v1 + np_v2}")
    print(f"Subtraction:          {np_v1 - np_v2}")
    print(f"Scalar multiplication: {3 * np_v1}")
    print()
    
    # Advanced operations
    print("Advanced Operations:")
    print(f"Dot product:          {np.dot(np_v1, np_v2):.1f}")
    print(f"Vector magnitude (v1): {np.linalg.norm(np_v1):.3f}")
    print(f"Vector magnitude (v2): {np.linalg.norm(np_v2):.3f}")
    print()


def demonstrate_complex_example() -> None:
    """Demonstrate more complex vector operations."""
    print("=" * 50)
    print("COMPLEX VECTOR OPERATIONS EXAMPLE")
    print("=" * 50)
    
    # More complex vectors
    v3 = np.array([1, -2, 3], dtype=float)
    v4 = np.array([2, 1, -1], dtype=float)
    
    print(f"Vector v3: {v3}")
    print(f"Vector v4: {v4}")
    print()
    
    # Combined operations
    print("Combined Operations:")
    print(f"v3 + v4 = {v3 + v4}")
    print(f"2*v3 - 3*v4 = {2*v3 - 3*v4}")
    print()
    
    # Additional metrics
    print("Vector Analysis:")
    print(f"||v3|| (magnitude): {np.linalg.norm(v3):.3f}")
    print(f"||v4|| (magnitude): {np.linalg.norm(v4):.3f}")
    print(f"v3 · v4 (dot product): {np.dot(v3, v4):.1f}")
    print()


def main() -> None:
    """Main function to run all demonstrations."""
    print("🧮 Day 1: Vector Operations Practice")
    print("Building foundations for linear algebra in machine learning")
    print()
    
    # Run all demonstrations
    demonstrate_basic_python()
    demonstrate_list_operations()
    demonstrate_numpy_operations()
    demonstrate_complex_example()
    
    print("=" * 50)
    print("✅ DAY 1 COMPLETE!")
    print("📚 Key Learnings:")
    print("   • Vector representation in Python")
    print("   • NumPy arrays vs Python lists")
    print("   • Basic vector arithmetic")
    print("   • Dot products and magnitudes")
    print("=" * 50)


if __name__ == "__main__":
    main()


