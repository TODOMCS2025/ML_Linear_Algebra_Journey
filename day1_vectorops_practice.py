x = 1

listNames = ["Alice", "Bob", "Charlie"] 

print(x)
print(listNames)

for name in listNames:
    print("Hello "+name+ ", Your welcome to the Python course!")

v1 = [1, 2, 3]
v2 = [4, 5, 6]

# Mathematical vector addition (element-wise)
def vector_add(a, b):
    """Add two vectors element by element"""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return [a[i] + b[i] for i in range(len(a))]

# Alternative using zip (more Pythonic)
def vector_add_zip(a, b):
    """Add two vectors using zip"""
    return [x + y for x, y in zip(a, b)]

# Vector addition examples
result_manual = vector_add(v1, v2)
result_zip = vector_add_zip(v1, v2)

print("Vector 1:", v1)
print("Vector 2:", v2)
print("Sum (manual):", result_manual)
print("Sum (zip):", result_zip)

# Mathematical explanation:
# v1 + v2 = [1, 2, 3] + [4, 5, 6] = [1+4, 2+5, 3+6] = [5, 7, 9]

# Using NumPy for more advanced vector operations
import numpy as np

print("\n--- NumPy Vector Operations ---")
np_v1 = np.array([1, 2, 3])
np_v2 = np.array([4, 5, 6])

# Vector addition with NumPy (much simpler!)
np_sum = np_v1 + np_v2
print("NumPy Vector 1:", np_v1)
print("NumPy Vector 2:", np_v2)
print("NumPy Sum:", np_sum)

# Other vector operations
print("Vector subtraction:", np_v1 - np_v2)
print("Scalar multiplication:", 3 * np_v1)
print("Dot product:", np.dot(np_v1, np_v2))
print("Vector magnitude (v1):", np.linalg.norm(np_v1))

# More complex example
v3 = np.array([1, -2, 3])
v4 = np.array([2, 1, -1])
print("\nComplex example:")
print("v3 =", v3)
print("v4 =", v4)
print("v3 + v4 =", v3 + v4)
print("2*v3 - 3*v4 =", 2*v3 - 3*v4)


