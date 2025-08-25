import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import matplotlib.pyplot as plt
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function: σ(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-z))

print(sigmoid(0))
print(sigmoid(100))
print(sigmoid(-100))

def hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Hypothesis function for logistic regression: h(X) = σ(X·θ)"""
    return sigmoid(X.dot(theta))    

def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute the cost for logistic regression using log-likelihood.
    Cost function: J(θ) = (-1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
    """
    m = len(y)
    h = hypothesis(X, theta)
    epsilon = 1e-15  # To avoid log(0)
    h_clipped = np.clip(h, epsilon, 1 - epsilon)
    
    # Correct logistic regression cost function
    cost = (-1/m) * np.sum(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
    return cost

#gradient descent function
def gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> Tuple[np.ndarray, list]:
    """Logistic regression gradient descent"""
    m, n = X.shape
    theta = np.zeros(n)  # Initialize theta for all features
    cost_history = []
    
    for i in range(iterations):
        # Make predictions using current parameters
        h = hypothesis(X, theta)
        
        # Calculate cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        # Calculate gradients
        gradient = (1/m) * X.T.dot(h - y)
        
        # Update parameters
        theta -= learning_rate * gradient
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")
    
    return theta, cost_history


# Generate synthetic data
np.random.seed(42)

mean0 = [1, 2]
cov0 = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(mean0, cov0, 100)
y0 = np.zeros(100)

mean1 = [5, 6]
cov1 = [[1, 0], [0, 1]]
X1 = np.random.multivariate_normal(mean1, cov1, 100)
y1 = np.ones(100)

X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

#Add the "intercept" term to X
X = np.hstack((np.ones((X.shape[0], 1)), X))

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")

# Train logistic regression
alpha = 0.01
iterations = 1000

print(f"\nTraining logistic regression...")
print(f"Learning rate: {alpha}, Iterations: {iterations}")

final_theta, cost_history = gradient_descent(X, y, alpha, iterations)

print(f"\nFinal parameters:")
print(f"θ₀ (bias): {final_theta[0]:.4f}")
print(f"θ₁ (feature1): {final_theta[1]:.4f}")
print(f"θ₂ (feature2): {final_theta[2]:.4f}")
print(f"Final cost: {cost_history[-1]:.6f}")


# Visualize the results
plt.figure(figsize=(15, 5))

# Plot 1: Data with decision boundary
plt.subplot(1, 3, 1)
plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color='red', label='Class 0', alpha=0.7)
plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color='blue', label='Class 1', alpha=0.7)

# Plot decision boundary: θ₀ + θ₁x₁ + θ₂x₂ = 0
# Solve for x₂: x₂ = -(θ₀ + θ₁x₁) / θ₂
x_values = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
y_values = -(final_theta[0] + final_theta[1] * x_values) / final_theta[2]
plt.plot(x_values, y_values, color='green', linewidth=2, label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Classification')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cost history
plt.subplot(1, 3, 2)
plt.plot(range(len(cost_history)), cost_history, color='purple', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Over Time')
plt.grid(True, alpha=0.3)

# Plot 3: Predictions
plt.subplot(1, 3, 3)
predictions = hypothesis(X, final_theta)
plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], c=predictions[y == 0], 
           cmap='Reds', label='Class 0', alpha=0.7, edgecolors='black')
plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c=predictions[y == 1], 
           cmap='Blues', label='Class 1', alpha=0.7, edgecolors='black')
plt.colorbar(label='Prediction Probability')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Prediction Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate accuracy
predictions_binary = (predictions >= 0.5).astype(int)
accuracy = np.mean(predictions_binary == y)
print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correctly classified: {np.sum(predictions_binary == y)}/{len(y)}")

# Show some predictions
print(f"\nSample predictions:")
for i in [0, 50, 100, 150]:
    prob = predictions[i]
    pred_class = predictions_binary[i]
    true_class = int(y[i])
    print(f"Point {i}: P(Class=1) = {prob:.4f}, Predicted: {pred_class}, True: {true_class}")