import numpy as np

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
