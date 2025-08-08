import numpy as np

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print("Array 1:", array1)
print("Array 2:", array2)
# Element-wise addition
result = np.add(array1, array2)
print("Element-wise addition result:", result)

result = np.subtract(array1, array2)
print("Element-wise subtraction result:", result)

result = np.multiply(array1, array2)
print("Element-wise multiplication result:", result)

result = np.divide(array1, array2)
print("Element-wise division result:", result)

result = np.dot(array1, array2)
print("Dot product result:", result)