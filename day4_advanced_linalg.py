### Day 4: Inverses and Determinants - Advanced Linear Algebra Practice
### Using NumPy for matrix inverses and determinants
### Floating-point precision issues and solutions
### Date: 2025-08-12

import numpy as np

matrix2X2 = np.array([[1, 2], [3, 4]])

result = np.linalg.det(matrix2X2)

print("Determinant of the 2x2 matrix:")
print(result)
# Inverse of a 2x2 matrix
matrix2X2_inv = np.linalg.inv(matrix2X2)
print("\nInverse of the 2x2 matrix:")
print(matrix2X2_inv)

unitmatrix = np.dot(matrix2X2_inv, matrix2X2)
print("\nUnit matrix (should be close to identity):")
print(unitmatrix)

print("\n=== FLOATING-POINT PRECISION EXPLANATION ===")
print("Notice that the result is NOT exactly:")
print("[[1, 0],")
print(" [0, 1]]")
print("\nInstead, we get tiny numbers like 1.11022302e-16 instead of 0")
print("This is due to FLOATING-POINT PRECISION ERRORS in computers")

print("\n--- SOLUTIONS ---")

# Solution 1: Round to reasonable decimal places
print("\n1. Rounding to reasonable precision:")
rounded_unit = np.round(unitmatrix, decimals=10)
print(rounded_unit)

# Solution 2: Use numpy's allclose function
print("\n2. Using np.allclose() to check if matrices are 'close enough':")
identity_2x2 = np.eye(2)  # Creates 2x2 identity matrix
print("Perfect identity matrix:")
print(identity_2x2)
print("Are they close enough?", np.allclose(unitmatrix, identity_2x2))

# Solution 3: Set very small numbers to zero
print("\n3. Setting tiny numbers to zero manually:")
clean_unit = np.where(np.abs(unitmatrix) < 1e-10, 0, unitmatrix)
print(clean_unit)

print("\n--- WHY THIS HAPPENS ---")
print("Computers store numbers in binary with limited precision")
print("When we do calculations like division and multiplication,")
print("tiny rounding errors accumulate")
print("1.11022302e-16 is approximately 0.000000000000000111")
print("That's 16 zeros after the decimal point - essentially zero!")

print("\n--- PRACTICAL APPROACH ---")
print("In practice, we:")
print("1. Accept that A⁻¹ × A ≈ I (approximately equal)")
print("2. Use np.allclose() to test equality")
print("3. Round results when displaying to users")

# Example with a more complex matrix
print("\n=== EXAMPLE WITH 3x3 MATRIX ===")
matrix3x3 = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
print("Original 3x3 matrix:")
print(matrix3x3)

det_3x3 = np.linalg.det(matrix3x3)
print(f"\nDeterminant: {det_3x3}")

if det_3x3 != 0:  # Matrix is invertible
    inv_3x3 = np.linalg.inv(matrix3x3)
    print("\nInverse:")
    print(inv_3x3)
    
    unit_3x3 = np.dot(inv_3x3, matrix3x3)
    print("\nA⁻¹ × A (raw result):")
    print(unit_3x3)
    
    print("\nA⁻¹ × A (rounded):")
    print(np.round(unit_3x3, 10))
    
    print("\nIs it close to identity?", np.allclose(unit_3x3, np.eye(3)))

#Solving equations
# 2x + 1y = 8
# 1x + 3y = 9
print("\n=== SOLVING LINEAR EQUATIONS ===")
# Coefficients matrix A and constants vector b
print("Solving the system of equations:")
print("2x + 1y = 8")
print("1x + 3y = 9")
# Coefficients matrix A and constants vector b

A = np.array([[2, 1], [1, 3]])
b = np.array([8, 9])

# Solving for [x, y]
solution = np.linalg.solve(A, b)
print("\nSolution for the system of equations:")
print(f"x = {solution[0]}, y = {solution[1]}")
# Verifying the solution
verification = np.dot(A, solution)
print("\nVerification (A @ solution):")
print(verification)
print("Expected b:", b)
# Verifying if the solution is correct
is_correct = np.allclose(verification, b)
print("Is the solution correct?", is_correct)
# This shows how to solve linear equations using matrix operations
# and verify the solution using NumPy's capabilities.