#create dataset of size and price of houses with realistic relationship
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(42)
from typing import Tuple
from numpy.linalg import inv, pinv
from numpy import ndarray
from mpl_toolkits.mplot3d import Axes3D

# Create realistic house data with linear relationship
n_samples = 30
np.random.seed(42)  # Ensure reproducible results
area = np.random.uniform(1000, 4000, n_samples)  # 1000-4000 sq ft

# Create price with known relationship: price = 150 * area + 50000 + noise
true_slope = 150
true_intercept = 50000
noise = np.random.normal(0, 20000, n_samples)
price = true_intercept + true_slope * area + noise

print(f"=== DATA GENERATION ===")
print(f"True relationship: Price = {true_intercept} + {true_slope} * Area + noise")
print(f"Generated {n_samples} samples")
print(f"Area range: {area.min():.0f} to {area.max():.0f} sq ft")
print(f"Price range: ${price.min():,.0f} to ${price.max():,.0f}")

#populate dictionary with area and price
house_data = {
    "area": area,
    "price": price
}

#explain the relationship between linear regression and gradient descent
# Linear Regression finds the best-fitting line through the data points by minimizing the cost function, which
# typically measures the difference between the predicted and actual values. Gradient Descent is an optimization algorithm
# used to minimize this cost function iteratively. It does so by calculating the gradient (or slope) of the cost function
# with respect to the model parameters (like slope and intercept in linear regression) and updating these parameters in the direction that reduces the cost.
# In essence, Gradient Descent is a method used to find the optimal parameters for a linear regression model.
# Gradient Descent is particularly useful when dealing with large datasets or multiple features, where calculating the optimal parameters analytically (as in the Normal Equation) can be computationally expensive.
#make the samples into a numpy array
X = np.array(list(house_data.values())).T
y = X[:, 1]  # Price
X = X[:, 0]  # Area
#reshape X to be a 2D array with one column
X = X.reshape(-1, 1)        
#add a column of ones to X for the intercept term
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term (intercept)
#hypothesis function
def hypothesis(X: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    return theta0 + theta1 * X
#gradient descent function
def gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> Tuple[float, float]:
    m = len(y)
    theta0, theta1 = 0.0, 0.0  # Initial parameters
    
    for i in range(iterations):
        # Make predictions using current parameters
        predictions = hypothesis(X.flatten(), theta0, theta1)
        
        # Calculate errors
        error = predictions - y
        
        # Calculate gradients
        grad_theta0 = (1/m) * np.sum(error)
        grad_theta1 = (1/m) * np.sum(error * X.flatten())
        
        # Update parameters
        theta0 -= learning_rate * grad_theta0
        theta1 -= learning_rate * grad_theta1
        
        # Print progress every 1000 iterations
        if i % 1000 == 0:
            cost = (1/(2*m)) * np.sum(error**2)
            print(f"Iteration {i}: Cost = {cost:.2f}")
    
    return theta0, theta1

# Example usage of gradient_descent with better learning rate
print(f"\n=== GRADIENT DESCENT TRAINING ===")
learning_rate = 0.00001  # Adjusted for unscaled data
iterations = 10000
print(f"Learning rate: {learning_rate}")
print(f"Iterations: {iterations}")

theta0, theta1 = gradient_descent(X, y, learning_rate, iterations)
print(f"\nFinal Parameters from Gradient Descent:")
print(f"θ₀ (intercept): {theta0:.2f} (true: {true_intercept})")
print(f"θ₁ (slope): {theta1:.6f} (true: {true_slope})")

# Calculate final cost
final_predictions = hypothesis(X.flatten(), theta0, theta1)
final_cost = (1/(2*len(y))) * np.sum((final_predictions - y)**2)
print(f"Final cost: {final_cost:.2f}")
# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data", alpha=0.7)
plt.plot(X, hypothesis(X.flatten(), theta0, theta1), color="red", label="Gradient Descent Line", linewidth=2)

# Add true line for comparison
X_line = np.linspace(X.min(), X.max(), 100)
true_line = true_intercept + true_slope * X_line
plt.plot(X_line, true_line, color="green", label="True Relationship", linestyle="--", linewidth=2)

plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Prices: Gradient Descent vs True Relationship")
# Format y-axis to show prices in thousands
plt.ticklabel_format(style='plain', axis='y')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# Standardize features function
def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, dict]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, {'mean': mean, 'std': std}
# Example usage of standardize_features with consistent data
print(f"\n=== FEATURE SCALING DEMONSTRATION ===")
# Use the same area data for consistency
X_multi = np.column_stack([area[:5], [3, 4, 3, 5, 4]])  # area + bedrooms for first 5 samples
X_scaled_multi, stats = standardize_features(X_multi)

print("Original data (first 5 samples):")
print(f"{'Sample':<8} {'Area (sq ft)':<12} {'Bedrooms':<10}")
for i, row in enumerate(X_multi):
    print(f"Row {i+1:<4}: {row[0]:>10.0f} {row[1]:>10.0f}")

print("\nScaled data:")
print(f"{'Sample':<8} {'Area (scaled)':<14} {'Bedrooms (scaled)':<16}")
for i, row in enumerate(X_scaled_multi):
    print(f"Row {i+1:<4}: {row[0]:>12.2f} {row[1]:>14.2f}")

# Show scaling statistics
print(f"\nScaling statistics:")
print(f"Area: mean={stats['mean'][0]:,.0f}, std={stats['std'][0]:,.0f}")
print(f"Bedrooms: mean={stats['mean'][1]:.1f}, std={stats['std'][1]:.1f}")
# Illustrate the importance of feature scaling in gradient descent
# Imagine making sandwiches where one ingredient is in grams and another in milligrams.
# If we don't scale the ingredients, our "gradient descent" (the process of making the sandwich)
# might take huge, erratic steps trying to balance the two ingredients, leading to a poorly made sandwich.
# However, if we scale both ingredients to a similar range, our steps become
# consistent and predictable, leading to a much better sandwich (and model)!

print("\n=== DEMONSTRATING FEATURE SCALING IMPORTANCE ===")

# Scale our single feature (area) for fair comparison
X_scaled_single, scaling_stats = standardize_features(X)

print(f"\nData transformation:")
print(f"Original X range: {X.min():,.0f} to {X.max():,.0f} sq ft")
print(f"Scaled X range: {X_scaled_single.min():.2f} to {X_scaled_single.max():.2f}")
print(f"Scaling: mean={scaling_stats['mean'][0]:,.0f}, std={scaling_stats['std'][0]:,.0f}")

# Run gradient descent on scaled data with higher learning rate
print("\n🔬 Training on SCALED data...")
learning_rate_scaled = 0.1  # Much higher learning rate possible with scaled data
iterations_scaled = 5000
theta0_scaled, theta1_scaled = gradient_descent(X_scaled_single, y, learning_rate_scaled, iterations_scaled)

print(f"\nTraining comparison:")
print(f"Unscaled data: learning_rate={learning_rate}, iterations={iterations}")
print(f"Scaled data: learning_rate={learning_rate_scaled}, iterations={iterations_scaled}")

print(f"\nResults comparison:")
print(f"Unscaled: θ₀={theta0:.2f}, θ₁={theta1:.6f}")
print(f"Scaled: θ₀={theta0_scaled:.2f}, θ₁={theta1_scaled:.2f}")

# Test predictions on same house
test_area = 2500
test_area_scaled = (test_area - scaling_stats['mean'][0]) / scaling_stats['std'][0]

pred_unscaled = hypothesis(np.array([test_area]), theta0, theta1)[0]
pred_scaled = hypothesis(test_area_scaled, theta0_scaled, theta1_scaled)

print(f"\nPrediction test for {test_area:,} sq ft house:")
print(f"Unscaled model: ${pred_unscaled:,.2f}")
print(f"Scaled model: ${pred_scaled:,.2f}")
print(f"True value: ${true_intercept + true_slope * test_area:,.2f}")
print(f"Prediction difference: ${abs(pred_unscaled - pred_scaled):,.2f}")

print(f"\nAccuracy comparison to true values:")
unscaled_error_intercept = abs(theta0 - true_intercept) / true_intercept * 100
unscaled_error_slope = abs(theta1 - true_slope) / true_slope * 100
print(f"Unscaled errors: θ₀={unscaled_error_intercept:.1f}%, θ₁={unscaled_error_slope:.1f}%")

print("\n✅ Feature scaling allows:")
print("• Higher learning rates")
print("• Faster convergence")
print("• More stable training")
print("• Better numerical stability")

print("\n=== COMPARISON WITH NORMAL EQUATION ===")

# Solve using Normal Equation for comparison
def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal equation: θ = (X^T X)^-1 X^T y"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta_optimal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_optimal

theta_optimal = normal_equation(X, y)

print(f"\nMethod Comparison:")
print(f"True values: θ₀={true_intercept:,.0f}, θ₁={true_slope}")
print(f"Normal Equation: θ₀={theta_optimal[0]:,.2f}, θ₁={theta_optimal[1]:.6f}")
print(f"Gradient Descent: θ₀={theta0:,.2f}, θ₁={theta1:.6f}")

# Calculate accuracy for each method
normal_error_intercept = abs(theta_optimal[0] - true_intercept) / true_intercept * 100
normal_error_slope = abs(theta_optimal[1] - true_slope) / true_slope * 100
gradient_error_intercept = abs(theta0 - true_intercept) / true_intercept * 100
gradient_error_slope = abs(theta1 - true_slope) / true_slope * 100

print(f"\nAccuracy comparison:")
print(f"Normal Equation errors: θ₀={normal_error_intercept:.2f}%, θ₁={normal_error_slope:.2f}%")
print(f"Gradient Descent errors: θ₀={gradient_error_intercept:.2f}%, θ₁={gradient_error_slope:.2f}%")

# Test prediction accuracy
test_area_pred = 2500
true_price = true_intercept + true_slope * test_area_pred
normal_pred = hypothesis(np.array([test_area_pred]), theta_optimal[0], theta_optimal[1])[0]
gradient_pred = hypothesis(np.array([test_area_pred]), theta0, theta1)[0]

print(f"\nPrediction test for {test_area_pred:,} sq ft house:")
print(f"True price: ${true_price:,.2f}")
print(f"Normal Equation: ${normal_pred:,.2f} (error: ${abs(normal_pred - true_price):,.2f})")
print(f"Gradient Descent: ${gradient_pred:,.2f} (error: ${abs(gradient_pred - true_price):,.2f})")

print("\n🎯 Key Takeaways:")
print("• Normal Equation: Direct analytical solution")
print("• Gradient Descent: Iterative optimization")
print("• Both should converge to similar results")
print("• Gradient Descent scales better with large datasets")
print("• Feature scaling is crucial for Gradient Descent performance")