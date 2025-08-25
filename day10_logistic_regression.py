# Day 4: Logistic Regression - Hour 1: Proving Our Old Tools Are Wrong for the Job
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression

print("="*70)
print("Hour 1: Proving Our Old Tools Are Wrong for the Job")
print("="*70)

print("\n=== STEP 1: REQUIRED READING/VIEWING (15 mins) ===")
print("📺 Watch Andrew Ng's video on 'Classification'")
print("🔍 Search: 'Andrew Ng Classification' on YouTube")
print("🎯 Key insight: Linear regression is a poor choice for classification")
print("✅ Ready to see WHY with code and visuals!")

print("\n=== STEP 2: CODE EXERCISE - THE FAILED ATTEMPT (25 mins) ===")

# Create a simple, artificial classification dataset
print("🧪 Creating simple tumor classification dataset...")
X = np.array([1, 2, 3, 5, 6, 7]).reshape(-1, 1)  # Tumor sizes
y = np.array([0, 0, 0, 1, 1, 1])  # 0=benign, 1=malignant

print("Dataset created:")
print("Tumor sizes (X):", X.flatten())
print("Labels (y):     ", y)
print("0 = Benign (small tumors)")
print("1 = Malignant (large tumors)")

# Import our normal equation from previous days
def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal equation: θ = (X^T X)^-1 X^T y"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta_optimal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_optimal

# Use linear regression on classification data (this will fail!)
print("\n🚨 Attempting LINEAR REGRESSION on classification data...")
theta_linear = normal_equation(X, y)

print(f"Linear regression parameters:")
print(f"θ₀ (intercept): {theta_linear[0]:.3f}")
print(f"θ₁ (slope): {theta_linear[1]:.3f}")

# Make predictions with linear regression
def linear_hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Linear hypothesis: h(x) = θ₀ + θ₁x"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

predictions_linear = linear_hypothesis(X, theta_linear)
print(f"\nLinear regression predictions: {predictions_linear}")
print("🚨 Notice: Some predictions are > 1 or < 0!")

print("\n=== STEP 3: VISUALIZATION AND ANALYSIS (20 mins) ===")

# Create the visualization
plt.figure(figsize=(12, 8))

# Plot the data points
plt.subplot(2, 2, 1)
plt.scatter(X[y==0], y[y==0], color='blue', s=100, label='Benign (0)', marker='o')
plt.scatter(X[y==1], y[y==1], color='red', s=100, label='Malignant (1)', marker='s')

# Plot the linear regression line
X_line = np.linspace(0, 8, 100)
X_line_b = np.c_[np.ones((len(X_line), 1)), X_line.reshape(-1, 1)]
y_line = X_line_b.dot(theta_linear)
plt.plot(X_line, y_line, color='green', linewidth=2, label='Linear Regression')

# Add horizontal lines at y=0 and y=1
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='purple', linestyle=':', alpha=0.7, label='Decision Boundary (0.5)')

plt.xlabel('Tumor Size')
plt.ylabel('Prediction')
plt.title('LINEAR REGRESSION FAILS for Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 1.5)

# Show the problems with specific examples
plt.subplot(2, 2, 2)
# Test points including problematic ones
test_points = np.array([0.5, 2.5, 4.5, 8]).reshape(-1, 1)
test_predictions = linear_hypothesis(test_points, theta_linear)

plt.scatter(X[y==0], y[y==0], color='blue', s=100, label='Benign (0)', marker='o')
plt.scatter(X[y==1], y[y==1], color='red', s=100, label='Malignant (1)', marker='s')
plt.plot(X_line, y_line, color='green', linewidth=2, label='Linear Regression')

# Highlight problematic predictions
for i, (point, pred) in enumerate(zip(test_points.flatten(), test_predictions)):
    color = 'orange' if pred < 0 or pred > 1 else 'purple'
    plt.scatter(point, pred, color=color, s=150, marker='x')
    plt.annotate(f'({point}, {pred:.2f})', 
                xy=(point, pred), xytext=(10, 10), 
                textcoords='offset points', fontsize=8)

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Tumor Size')
plt.ylabel('Prediction')
plt.title('PROBLEMS: Predictions < 0 and > 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 1.5)

# Demonstrate outlier effect
plt.subplot(2, 2, 3)
# Add an outlier
X_outlier = np.array([1, 2, 3, 5, 6, 7, 15]).reshape(-1, 1)  # Added large tumor
y_outlier = np.array([0, 0, 0, 1, 1, 1, 1])

# Fit linear regression with outlier
theta_outlier = normal_equation(X_outlier, y_outlier)
y_line_outlier = X_line_b.dot(theta_outlier)

# Plot original and outlier-affected lines
plt.scatter(X[y==0], y[y==0], color='blue', s=100, label='Original Benign', marker='o')
plt.scatter(X[y==1], y[y==1], color='red', s=100, label='Original Malignant', marker='s')
plt.scatter([15], [1], color='darkred', s=200, label='Outlier (Large Malignant)', marker='*')

plt.plot(X_line, y_line, color='green', linewidth=2, label='Original Line', linestyle='--')
plt.plot(X_line, y_line_outlier, color='orange', linewidth=3, label='Line with Outlier')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='purple', linestyle=':', alpha=0.7)

plt.xlabel('Tumor Size')
plt.ylabel('Prediction')
plt.title('OUTLIER EFFECT: Line Shifts Dramatically')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 1.5)
plt.xlim(0, 16)

# Show what we need instead
plt.subplot(2, 2, 4)
plt.scatter(X[y==0], y[y==0], color='blue', s=100, label='Benign (0)', marker='o')
plt.scatter(X[y==1], y[y==1], color='red', s=100, label='Malignant (1)', marker='s')

# Show sigmoid curve (preview of logistic regression)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Fit a rough sigmoid for demonstration
z_line = 2 * (X_line - 4)  # Rough approximation
sigmoid_line = sigmoid(z_line)
plt.plot(X_line, sigmoid_line, color='purple', linewidth=3, label='Sigmoid (What We Need)')

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='purple', linestyle=':', alpha=0.7)

plt.xlabel('Tumor Size')
plt.ylabel('Probability')
plt.title('SOLUTION: Sigmoid Always Between 0 and 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ANALYSIS: Critical Problems with Linear Regression for Classification")
print("="*70)

print("""
🔍 ANALYSIS QUESTIONS AND ANSWERS:

1. Q: Does the line produce predictions greater than 1 or less than 0?
   A: YES! Linear regression predictions can be any real number.
   
   Examples from our model:
   - For tumor size 0.5: prediction = {:.3f} (< 0 ❌)
   - For tumor size 8.0: prediction = {:.3f} (> 1 ❌)
   
2. Q: Why is this a problem for predicting yes/no?
   A: Classification outputs should be PROBABILITIES (0 to 1).
   - Negative predictions make no sense for probability
   - Predictions > 1 violate probability axioms
   - We need: P(malignant) ∈ [0, 1]

3. Q: What happens with outliers?
   A: The line shifts dramatically! One large tumor at size 15 
   completely changes predictions for ALL other points.
   - Original θ₁ = {:.3f}
   - With outlier θ₁ = {:.3f}
   - This makes the model unreliable and unstable

🎯 KEY TAKEAWAY:
""".format(
    linear_hypothesis(np.array([[0.5]]), theta_linear)[0],
    linear_hypothesis(np.array([[8.0]]), theta_linear)[0],
    theta_linear[1],
    theta_outlier[1]
))

print("WE NEED A MODEL THAT ALWAYS OUTPUTS A VALUE BETWEEN 0 AND 1")

print("""
🚨 WHY LINEAR REGRESSION FAILS FOR CLASSIFICATION:

❌ Range Problem: Outputs can be (-∞, +∞), not [0, 1]
❌ Outlier Sensitivity: Single extreme point shifts entire line
❌ No Probability Interpretation: Negative "probabilities" are meaningless
❌ Poor Decision Boundary: Linear threshold doesn't capture sigmoid nature
❌ Optimization Issues: MSE loss not ideal for binary outcomes

✅ WHAT WE NEED (Coming Next):
• Sigmoid function: Maps any input to [0, 1]
• Logistic regression: Natural for binary classification
• Maximum Likelihood Estimation: Proper loss function
• Robust to outliers: Bounded output space
• Probabilistic interpretation: P(y=1|x)

🏆 EDUCATIONAL VALUE:
This isn't wasted effort! By seeing how a tool FAILS, we gain deep 
understanding of why the correct tool (logistic regression) is designed 
the way it is. We now appreciate the sigmoid function's brilliance!
""")

print("\n🎯 NEXT: We'll implement the CORRECT tool - Logistic Regression!")
print("Hour 2: The Sigmoid Function - Nature's Perfect S-Curve")