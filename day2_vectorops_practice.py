### Day 2: Core Vector Operations with Numpy - Vector Operations Practice
### Using NumPy arrays for vector operations
### Element-wise operations, dot product, norms
### Date: 2025-08-08
import numpy as np

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print("Array 1:", array1)
print("Array 2:", array2)
# Element-wise addition
result = np.add(array1, array2)
print("Element-wise addition result:", result)

# Element-wise subtraction, multiplication, and division
result = np.subtract(array1, array2)
print("Element-wise subtraction result:", result)

result = np.multiply(array1, array2)
print("Element-wise multiplication result:", result)

result = np.divide(array1, array2)
print("Element-wise division result:", result)

# Element-wise dot product
result = np.dot(array1, array2)
print("Dot product result:", result)

# Element-wise dot product using @ operator
result = array1 @ array2
print("Dot product using @ operator:", result)

#L2 Norm
array1 = np.array([5, 3, 4])
norm = (5**2 + 3**2 + 4**2)**0.5
print("L2 Norm:", norm)

norm = np.linalg.norm(array1)
print("L2 Norm using np.linalg.norm:", norm)

#L1 Norm
array1 = np.array([5, 3, 4])
norm = abs(5) + abs(3) + abs(4)
print("L1 Norm:", norm)

#Squared L2 Norm
array1 = np.array([5, 3, 4])
squared_norm = (5**2 + 3**2 + 4**2)
print("Squared L2 Norm:", squared_norm)
print("Squared L2 Norm using np.linalg.norm:", np.linalg.norm(array1)**2)
dot = np.dot(array1, array1)
print("Squared L2 Norm using np.dot:", dot)