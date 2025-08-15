"""
Day 3: Core Matrix Operations with Numpy - Matrix Operations Practice

This module introduces basic matrix operations using NumPy arrays. It covers:
- Element-wise operations
- Matrix multiplication
- Determinant and inverse calculations
Learning objectives:
- Understanding numpy arrays for matrix representation
- Performing element-wise operations with numpy
- Calculating matrix multiplication and transposition
- Determinant and inverse calculations using numpy

Date: 2025-08-11
"""

import numpy as np

def demonstrate_matrix2X2_operations():
    """Demonstrate basic matrix operations using NumPy."""
    print("=" * 50)
    print("MATRIX OPERATIONS DEMONSTRATION")
    print("=" * 50)

    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])

    print("Array 1:", matrix1)
    print("Array 2:", matrix2)
    # Element-wise addition
    result = np.add(matrix1, matrix2)

    print("Element-wise addition result:", result)

    # Element-wise subtraction, multiplication, and division
    result = np.subtract(matrix1, matrix2)
    print("Element-wise subtraction result:", result)

    result = np.multiply(matrix1, matrix2)
    print("Element-wise multiplication result:", result)

    result = np.divide(matrix1, matrix2)
    print("Element-wise division result:", result)

def demonstrate_matrix3X3_operations():
    """Demonstrate basic 3x3 matrix operations using NumPy."""
    print("\n" + "=" * 50)
    print("3x3 MATRIX OPERATIONS DEMONSTRATION")
    print("=" * 50)
    matrixa = np.array([[1,0,-1],[2,4,2],[3,5,3]])
    matrixb = np.array([[3,4,5],[1,2,3],[0,1,0]])
    # Element-wise addition of matrices
    result = np.add(matrixa, matrixb)
    print("Element-wise addition of matrices result:", result)
    # Element-wise subtraction of matrices
    result = np.subtract(matrixa, matrixb)
    print("Element-wise subtraction of matrices result:", result)
    # Element-wise multiplication of matrices
    resultM = np.multiply(matrixa, matrixb)
    print("Element-wise multiplication of matrices result:", resultM)
    # Element-wise division of matrices
    result = np.divide(matrixa, matrixb)
    print("Element-wise division of matrices result:", result)

    print("\n=== MATRIX MULTIPLICATION EXPLANATION ===")
    print("Matrix A:")
    print(matrixa)
    print("Matrix B:")
    print(matrixb)

    print("\n--- 1. ELEMENT-WISE MULTIPLICATION (Hadamard Product) ---")
    element_wise = np.multiply(matrixa, matrixb)
    print("Element-wise multiplication (A ⊙ B):")
    print(element_wise)
    print("Formula: result[i,j] = A[i,j] * B[i,j]")
    print("Example: result[0,0] = 1 * 3 =", element_wise[0,0])
    print("Example: result[1,1] = 4 * 2 =", element_wise[1,1])

    print("\n--- 2. MATRIX MULTIPLICATION (Dot Product) ---")
    matrix_mult = np.dot(matrixa, matrixb)
    print("Matrix multiplication (A × B):")
    print(matrix_mult)

    print("\nHow matrix multiplication works:")
    print("For result[i,j], we take row i from A and column j from B")
    print("Then multiply corresponding elements and sum them up")

    print("\nStep by step for result[0,0]:")
    print("Row 0 of A: [1, 0, -1]")
    print("Column 0 of B: [3, 1, 0]")
    print("Calculation: (1×3) + (0×1) + (-1×0) = 3 + 0 + 0 = 3")

    print("\nStep by step for result[0,1]:")
    print("Row 0 of A: [1, 0, -1]")
    print("Column 1 of B: [4, 2, 1]")
    print("Calculation: (1×4) + (0×2) + (-1×1) = 4 + 0 - 1 = 3")

    print("\nStep by step for result[1,0]:")
    print("Row 1 of A: [2, 4, 2]")
    print("Column 0 of B: [3, 1, 0]")
    print("Calculation: (2×3) + (4×1) + (2×0) = 6 + 4 + 0 = 10")

    print("\n--- COMPLETE MATRIX MULTIPLICATION BREAKDOWN ---")
    for i in range(3):
        for j in range(3):
            row_a = matrixa[i, :]
            col_b = matrixb[:, j]
            result_val = np.dot(row_a, col_b)
            print(f"result[{i},{j}]: {row_a} · {col_b} = {result_val}")

    print("\n--- VERIFICATION ---")
    print("Manual calculation matches numpy result:")
    print("Our manual result:", matrix_mult)
    print("NumPy @ operator:", matrixa @ matrixb)  # Alternative syntax
    print("Are they equal?", np.array_equal(matrix_mult, matrixa @ matrixb))


def main():
    """Main function to run demonstrations."""
    demonstrate_matrix2X2_operations()
    demonstrate_matrix3X3_operations()

    print("=" * 50)
    print("✅ DAY 3 COMPLETE!")
    print("📚 Key Learnings:")
    print("   • Matrix operations")
    print("   • Element-wise vs matrix multiplication")
    print("=" * 50) 

if __name__ == "__main__":
    main()