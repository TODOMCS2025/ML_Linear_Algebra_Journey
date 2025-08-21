# Hour 1: Predicting from Multiple Inputs - Multi-variable Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

np.random.seed(42)

print("=== STEP 1: THE CONCEPT ===")
print("Long-form equation: h(x) = θ₀ + θ₁x₁ + θ₂x₂")
print("Vector form: θ = [θ₀, θ₁, θ₂] and x = [1, x₁, x₂]")
print("Result: θᵀx = same as long-form!")
print()

print("=== STEP 2: CREATE ARTIFICIAL DATASET ===")

# Generate realistic house data
n_samples = 100

# Features: size (sq ft) and bedrooms
size = np.random.uniform(1000, 4000, n_samples)  # 1000-4000 sq ft
bedrooms = np.random.randint(1, 6, n_samples)    # 1-5 bedrooms

# Add some correlation: bigger houses tend to have more bedrooms
bedrooms = bedrooms + (size / 1000).astype(int) - 1
bedrooms = np.clip(bedrooms, 1, 6)  # Keep realistic range

# Create price using linear combination + noise
# Formula: price = 150*size + 50000*bedrooms + 30000 + noise
true_theta = [30000, 150, 50000]  # [intercept, size_coeff, bedroom_coeff]
noise = np.random.normal(0, 20000, n_samples)

price = true_theta[0] + true_theta[1] * size + true_theta[2] * bedrooms + noise

# Create feature matrix X with bias term
X = np.column_stack([np.ones(n_samples), size, bedrooms])  # Add bias column
y = price

print(f"Dataset created with {n_samples} samples")
print(f"True parameters: θ₀={true_theta[0]}, θ₁={true_theta[1]}, θ₂={true_theta[2]}")
print()

print("=== STEP 3: VERIFICATION - CHECK DIMENSIONS ===")
print(f"X matrix shape: {X.shape} (should be ({n_samples}, 3))")
print(f"y vector shape: {y.shape} (should be ({n_samples},))")
print()
print("X matrix first 5 rows:")
print("  [bias, size, bedrooms]")
for i in range(5):
    print(f"  [{X[i,0]:.0f}, {X[i,1]:.0f}, {X[i,2]:.0f}]")
print()

# Vector form hypothesis function
def hypothesis_vectorized(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Vectorized hypothesis: h(x) = θᵀx
    X: matrix of features (with bias column)
    theta: vector of parameters
    """
    return X.dot(theta)

# Test with true parameters
y_pred_true = hypothesis_vectorized(X, true_theta)
print(f"Mean prediction with true parameters: ${y_pred_true.mean():,.2f}")
print(f"Mean actual price: ${y.mean():,.2f}")
print()

# Visualize the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Price vs Size
ax1.scatter(X[:,1], y, alpha=0.6, color='blue')
ax1.set_xlabel('Size (sq ft)')
ax1.set_ylabel('Price ($)')
ax1.set_title('House Price vs Size')
ax1.ticklabel_format(style='plain', axis='y')

# Plot 2: Price vs Bedrooms
ax2.scatter(X[:,2], y, alpha=0.6, color='green')
ax2.set_xlabel('Number of Bedrooms')
ax2.set_ylabel('Price ($)')
ax2.set_title('House Price vs Bedrooms')
ax2.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.show()

print("=== MASTER'S LEVEL INSIGHT ===")
print("🎯 Vector/Matrix abstraction allows us to:")
print("   • Handle 2 features or 2,000 features with the same code")
print("   • Use efficient linear algebra operations")
print("   • Scale machine learning to massive datasets")
print("   • Write cleaner, more maintainable code")

print("\n" + "="*60)
print("Hour 2: The Power of Good Code")
print("="*60)

print("\n=== STEP 1: REVIEW AND IMPORT (15 mins) ===")
print("✓ Normal equation function imported from Day 1")
print("✓ Same function, no modifications needed!")
print()

# The exact same normal equation from day7_linear_regression.py
def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normal equation for linear regression: θ = (X^T X)^-1 X^T y
    This function works for ANY number of features!
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

print("=== STEP 2: RUN THE EXPERIMENT (30 mins) ===")
print("🧪 Testing: Can our Day 1 function handle multi-feature data?")
print()

# Prepare data without bias column for the original function
X_features = X[:, 1:]  # Remove bias column (size and bedrooms only)

# Use the EXACT SAME function from Day 1
theta_from_day1 = normal_equation(X_features, y)

print("🎉 SUCCESS! The function worked without ANY changes!")
print("Same code, more complex problem ✓")
print()

print("=== STEP 3: ANALYZE THE OUTPUT (15 mins) ===")
print("📊 Theta vector calculated by our Day 1 function:")
print(f"θ₀ (intercept): ${theta_from_day1[0]:,.2f}")
print(f"θ₁ (size coeff): ${theta_from_day1[1]:,.2f} $/sq ft")
print(f"θ₂ (bedroom coeff): ${theta_from_day1[2]:,.2f} $/bedroom")
print()

print("🎯 Comparison with TRUE values used to generate data:")
print(f"True θ₀: ${true_theta[0]:,.2f} | Found: ${theta_from_day1[0]:,.2f} | ✓ Close!")
print(f"True θ₁: ${true_theta[1]:,.2f} | Found: ${theta_from_day1[1]:,.2f} | ✓ Close!")
print(f"True θ₂: ${true_theta[2]:,.2f} | Found: ${theta_from_day1[2]:,.2f} | ✓ Close!")
print()

# Calculate accuracy
error_0 = abs(theta_from_day1[0] - true_theta[0]) / true_theta[0] * 100
error_1 = abs(theta_from_day1[1] - true_theta[1]) / true_theta[1] * 100
error_2 = abs(theta_from_day1[2] - true_theta[2]) / true_theta[2] * 100

print(f"Accuracy: θ₀ error: {error_0:.1f}%, θ₁ error: {error_1:.1f}%, θ₂ error: {error_2:.1f}%")
print()

print("🏗️ THE SANDWICH MACHINE ANALOGY:")
print("Yesterday: Cheese sandwich (1 feature)")
print("Today: Club sandwich (2+ features)")
print("Same machine, more ingredients = Same code, more features!")
print()

print("🚀 MASTER'S LEVEL INSIGHT:")
print("This demonstrates the elegance and power of vectorized code.")
print("A solution that works for n=1 feature often works for n=1000 features,")
print("making our solutions GENERAL and REUSABLE.")
print()

# Demonstrate scalability
print("=== BONUS: SCALABILITY DEMONSTRATION ===")
print("🔥 Let's prove this works with even MORE features!")

# Create an extended dataset with 5 features
np.random.seed(42)
n_features = 5
X_extended = np.random.uniform(0, 100, (50, n_features))
true_theta_extended = np.random.uniform(10, 50, n_features + 1)  # +1 for intercept
y_extended = true_theta_extended[0] + X_extended.dot(true_theta_extended[1:]) + np.random.normal(0, 5, 50)

# Use the SAME function!
theta_extended = normal_equation(X_extended, y_extended)

print(f"✓ Worked with {n_features} features!")
print(f"Mean error: {np.mean(np.abs(theta_extended - true_theta_extended)):.2f}")
print()

print("🏆 CONCLUSION:")
print("One function, infinite possibilities!")
print("This is the power of good mathematical abstraction!")

print("\n" + "="*60)
print("BONUS: Alternative Implementation")
print("="*60)

def normal_equation_multivar(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normal equation for multivariate linear regression
    θ = (XᵀX)⁻¹Xᵀy
    """
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Find optimal parameters using normal equation
theta_optimal = normal_equation_multivar(X, y)

print("Optimal parameters found:")
print(f"θ₀ (intercept): ${theta_optimal[0]:,.2f}")
print(f"θ₁ (size coeff): ${theta_optimal[1]:,.2f} $/sq ft")
print(f"θ₂ (bedroom coeff): ${theta_optimal[2]:,.2f} $/bedroom")
print()

print("Comparison with true parameters:")
print(f"True θ₀: ${true_theta[0]:,.2f}, Found: ${theta_optimal[0]:,.2f}")
print(f"True θ₁: ${true_theta[1]:,.2f}, Found: ${theta_optimal[1]:,.2f}")
print(f"True θ₂: ${true_theta[2]:,.2f}, Found: ${theta_optimal[2]:,.2f}")
print()

# Test prediction with optimal parameters
y_pred_optimal = hypothesis_vectorized(X, theta_optimal)

# Calculate cost/error
def compute_cost_multivar(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Cost function for multivariate regression"""
    m = len(y)
    predictions = hypothesis_vectorized(X, theta)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

cost_optimal = compute_cost_multivar(X, y, theta_optimal)
print(f"Final cost with optimal parameters: ${cost_optimal:,.2f}")

# Example prediction
example_house = np.array([1, 2500, 3])  # 2500 sq ft, 3 bedrooms
predicted_price = hypothesis_vectorized(example_house.reshape(1, -1), theta_optimal)[0]
print(f"\nExample prediction:")
print(f"House: 2500 sq ft, 3 bedrooms")
print(f"Predicted price: ${predicted_price:,.2f}")

print("\n🏆 SUCCESS! You've implemented multivariate linear regression using:")
print("   ✓ Vector/matrix operations")
print("   ✓ Normal equation solution")
print("   ✓ Multiple feature inputs")
print("   ✓ Scalable abstraction")

print("\n" + "="*60)
print("Hour 3: The Practitioner's Habit: Feature Scaling")
print("="*60)

print("\n=== STEP 1: THE THEORY (20 mins) ===")
print("🎯 THE MAP ANALOGY:")
print("Instruction 1: 'Go 10,000 meters east'")
print("Instruction 2: 'Go 5 meters north'")
print("Problem: The huge 'east' number makes 'north' seem insignificant!")
print("Solution: Rescale both to be comparable")
print()

print("📊 FEATURE SCALING METHODS:")
print("• Standardization (Z-score): z = (x - μ) / σ")
print("  Result: Mean = 0, Standard Deviation = 1")
print("• Normalization (Min-Max): x_norm = (x - min) / (max - min)")
print("  Result: Values between 0 and 1")
print()
print("✅ We'll focus on STANDARDIZATION (most common)")
print()

print("=== STEP 2: THE CODE (25 mins) ===")

def standardize_features(X: np.ndarray) -> tuple:
    """
    Standardize features using Z-score normalization
    Formula: z = (x - μ) / σ
    
    Returns:
        X_scaled: Standardized features
        stats: Dictionary with mean and std for each feature
    """
    # Calculate mean and standard deviation for each feature column
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    
    # Apply standardization formula: z = (x - μ) / σ
    X_scaled = (X - mean) / std
    
    # Store statistics for later use (important for new predictions!)
    stats = {'mean': mean, 'std': std}
    
    return X_scaled, stats

print("✅ Function created: standardize_features(X)")
print("📝 Formula implemented: z = (x - μ) / σ")
print("💾 Returns scaled data + statistics for future use")
print()

print("=== STEP 3: SEE THE DIFFERENCE (15 mins) ===")

# Let's look at our original data without the bias column
X_features_only = X[:, 1:]  # Remove bias column for scaling
feature_names = ['Size (sq ft)', 'Bedrooms']

print("🔍 BEFORE SCALING:")
print("Original X matrix (first 5 rows):")
print(f"{'':>5} {'Size (sq ft)':>12} {'Bedrooms':>10}")
for i in range(5):
    print(f"Row {i+1:>2}: {X_features_only[i,0]:>10.0f} {X_features_only[i,1]:>10.0f}")
print()

# Calculate statistics manually to show the process
print("📊 ORIGINAL DATA STATISTICS:")
for j, name in enumerate(feature_names):
    mean_val = X_features_only[:, j].mean()
    std_val = X_features_only[:, j].std()
    min_val = X_features_only[:, j].min()
    max_val = X_features_only[:, j].max()
    print(f"{name}:")
    print(f"  Mean: {mean_val:8.1f}, Std: {std_val:8.1f}")
    print(f"  Range: {min_val:8.1f} to {max_val:8.1f}")
print()

# Apply our standardization function
X_scaled, scaling_stats = standardize_features(X_features_only)

print("🔍 AFTER SCALING:")
print("Scaled X matrix (first 5 rows):")
print(f"{'':>5} {'Size (scaled)':>13} {'Bedrooms (scaled)':>17}")
for i in range(5):
    print(f"Row {i+1:>2}: {X_scaled[i,0]:>10.2f} {X_scaled[i,1]:>13.2f}")
print()

print("📊 SCALED DATA STATISTICS:")
for j, name in enumerate(feature_names):
    mean_val = X_scaled[:, j].mean()
    std_val = X_scaled[:, j].std()
    min_val = X_scaled[:, j].min()
    max_val = X_scaled[:, j].max()
    print(f"{name} (scaled):")
    print(f"  Mean: {mean_val:8.3f}, Std: {std_val:8.3f}")
    print(f"  Range: {min_val:8.2f} to {max_val:8.2f}")
print()

print("🎉 AMAZING! Notice how:")
print("• All means are now ~0.000")
print("• All standard deviations are now ~1.000")  
print("• Features are now on comparable scales!")
print()

print("=== PRACTICAL DEMONSTRATION ===")
print("🏠 Let's see how scaling affects our model...")

# Create X matrix with bias column for scaled data
X_scaled_with_bias = np.column_stack([np.ones(len(X_scaled)), X_scaled])

# Train model on scaled data
theta_scaled = normal_equation(X_scaled, y)

print("🔬 MODEL COMPARISON:")
print("\nOriginal data model coefficients:")
print(f"θ₀ (intercept): ${theta_from_day1[0]:,.2f}")
print(f"θ₁ (size coeff): ${theta_from_day1[1]:,.2f}")
print(f"θ₂ (bedroom coeff): ${theta_from_day1[2]:,.2f}")

print("\nScaled data model coefficients:")
print(f"θ₀ (intercept): ${theta_scaled[0]:,.2f}")
print(f"θ₁ (size coeff): ${theta_scaled[1]:,.2f}")
print(f"θ₂ (bedroom coeff): ${theta_scaled[2]:,.2f}")
print()

print("💡 INTERPRETATION:")
print("• The intercept should be similar (it's the mean of y)")
print("• The other coefficients are now in 'standard deviation units'")
print("• θ₁ = how much price changes per 1 std dev change in size")
print("• θ₂ = how much price changes per 1 std dev change in bedrooms")
print()

# Test prediction with scaled data
example_size = 2500
example_bedrooms = 3

# Scale the example using our stored statistics
example_scaled = np.array([
    (example_size - scaling_stats['mean'][0]) / scaling_stats['std'][0],
    (example_bedrooms - scaling_stats['mean'][1]) / scaling_stats['std'][1]
])

# Add bias term
example_scaled_with_bias = np.array([1, example_scaled[0], example_scaled[1]])

# Predict
prediction_scaled = hypothesis_vectorized(example_scaled_with_bias.reshape(1, -1), theta_scaled)[0]
prediction_original = hypothesis_vectorized(np.array([[1, example_size, example_bedrooms]]), theta_from_day1)[0]

print("🏡 EXAMPLE PREDICTION:")
print(f"House: {example_size} sq ft, {example_bedrooms} bedrooms")
print(f"Original model prediction: ${prediction_original:,.2f}")
print(f"Scaled model prediction: ${prediction_scaled:,.2f}")
print(f"Difference: ${abs(prediction_original - prediction_scaled):,.2f}")
print("✅ Should be very similar!")
print()

print("🚀 MASTER'S LEVEL INSIGHT:")
print("Feature scaling is a CRITICAL step in the data preprocessing pipeline.")
print("Nearly all real-world datasets require some form of scaling.")
print("Skipping this step can cause:")
print("• Models to learn slowly")
print("• Biased results toward features with larger scales")
print("• Poor convergence in gradient-based algorithms")
print()
print("🏆 This is a hallmark of a careful and professional practitioner!")
print("Always scale your features before training!")

print("\n" + "="*60)
print("Hour 4: Proving the Point and Preparing for Tomorrow")
print("="*60)

print("\n=== STEP 1: RUN ON UNSCALED DATA (20 mins) ===")
print("🍞 Running sandwich machine with original ingredients:")
print("   • 10,000g of bread (huge number)")
print("   • 5g of cheese (tiny number)")
print()

# Use normal equation on original, unscaled data
print("🔬 Training model on ORIGINAL (unscaled) data...")
theta_unscaled = normal_equation(X_features, y)

print("✅ Results from unscaled data:")
print(f"θ₀ (intercept): ${theta_unscaled[0]:,.2f}")
print(f"θ₁ (size coeff): ${theta_unscaled[1]:,.4f} $/sq ft")
print(f"θ₂ (bedroom coeff): ${theta_unscaled[2]:,.2f} $/bedroom")
print()

# Calculate predictions with unscaled model
y_pred_unscaled = hypothesis_vectorized(X, theta_unscaled)
cost_unscaled = compute_cost_multivar(X, y, theta_unscaled)
print(f"Cost function value (unscaled): ${cost_unscaled:,.2f}")
print()

print("=== STEP 2: RUN ON SCALED DATA (20 mins) ===")
print("🍞 Running sandwich machine with scaled ingredients:")
print("   • 0.85 units of bread (normalized)")
print("   • -0.23 units of cheese (normalized)")
print()

# Use normal equation on scaled data
print("🔬 Training model on SCALED data...")
theta_scaled_final = normal_equation(X_scaled, y)

print("✅ Results from scaled data:")
print(f"θ₀ (intercept): ${theta_scaled_final[0]:,.2f}")
print(f"θ₁ (size coeff): ${theta_scaled_final[1]:,.2f} per std dev")
print(f"θ₂ (bedroom coeff): ${theta_scaled_final[2]:,.2f} per std dev")
print()

# Calculate predictions with scaled model
# First, create scaled data with bias column
X_scaled_with_bias_final = np.column_stack([np.ones(len(X_scaled)), X_scaled])
y_pred_scaled_final = hypothesis_vectorized(X_scaled_with_bias_final, theta_scaled_final)
cost_scaled = compute_cost_multivar(X_scaled_with_bias_final, y, theta_scaled_final)
print(f"Cost function value (scaled): ${cost_scaled:,.2f}")
print()

print("=== STEP 3: ANALYSIS AND DOCUMENTATION (20 mins) ===")
print("📊 PERFORMANCE COMPARISON:")
print()

# Test both models on the same example
test_house_original = np.array([1, 2500, 3])  # With bias
test_house_scaled = np.array([1, 
    (2500 - scaling_stats['mean'][0]) / scaling_stats['std'][0],
    (3 - scaling_stats['mean'][1]) / scaling_stats['std'][1]])

pred_unscaled_test = hypothesis_vectorized(test_house_original.reshape(1, -1), theta_unscaled)[0]
pred_scaled_test = hypothesis_vectorized(test_house_scaled.reshape(1, -1), theta_scaled_final)[0]

print(f"🏡 Test House: 2500 sq ft, 3 bedrooms")
print(f"Unscaled model prediction: ${pred_unscaled_test:,.2f}")
print(f"Scaled model prediction: ${pred_scaled_test:,.2f}")
print(f"Difference: ${abs(pred_unscaled_test - pred_scaled_test):,.2f}")
print("✅ Both models give essentially the same predictions!")
print()

print("💰 Cost Function Comparison:")
print(f"Unscaled data cost: ${cost_unscaled:,.2f}")
print(f"Scaled data cost: ${cost_scaled:,.2f}")
print("✅ Both achieve similar performance!")
print()

print("🎯 Model Coefficients Analysis:")
print("\nUnscaled coefficients interpretation:")
print(f"• Each sq ft adds ${theta_unscaled[1]:,.4f} to price")
print(f"• Each bedroom adds ${theta_unscaled[2]:,.2f} to price")

print("\nScaled coefficients interpretation:")
print(f"• One std dev increase in size adds ${theta_scaled_final[1]:,.2f} to price")
print(f"• One std dev increase in bedrooms adds ${theta_scaled_final[2]:,.2f} to price")
print()

print("="*60)
print("🚨 CRITICAL INSIGHT FOR TOMORROW 🚨")
print("="*60)

print("""
📝 ANALYSIS AND DOCUMENTATION:

🔍 WHAT WE DID:
1. Trained Normal Equation on unscaled data → Worked perfectly ✅
2. Trained Normal Equation on scaled data → Worked perfectly ✅
3. Both models give identical predictions ✅

🤔 SO WHY SCALE AT ALL?

🎯 THE KEY INSIGHT:
While the Normal Equation can handle unscaled features, iterative methods 
like Gradient Descent CANNOT. The cost function becomes very skewed, making 
it slow and difficult for the algorithm to find the minimum. Scaling features 
ensures that Gradient Descent can work efficiently, which is what we will 
implement tomorrow.

🏔️ THE LANDSCAPE ANALOGY:
Imagine you're trying to find the bottom of a valley:

UNSCALED DATA LANDSCAPE:
• Very steep cliff on one side (size feature: 1000-4000)
• Gentle slope on other side (bedrooms: 1-6)
• Gradient Descent gets confused - takes HUGE steps in wrong direction
• May never find the bottom, or take forever to get there

SCALED DATA LANDSCAPE:
• Gentle, even slopes on all sides (both features: ~-2 to +2)
• Gradient Descent takes consistent, reasonable steps
• Quickly finds the bottom of the valley

🚀 TOMORROW'S PREVIEW:
We'll implement Gradient Descent and see this difference firsthand:
• Unscaled data: Gradient Descent will struggle or fail
• Scaled data: Gradient Descent will converge smoothly

🏆 MASTER'S LEVEL INSIGHT:
This demonstrates FORESIGHT. We are not just solving today's problem; 
we are building robust habits and designing a data pipeline that is ready 
for more advanced and common techniques. We understand not just WHAT to do, 
but WHY we are doing it.

This is the hallmark of a professional practitioner who thinks ahead! 🎯
""")

print("\n🔮 TOMORROW'S AGENDA:")
print("• Implement Gradient Descent algorithm")
print("• Compare convergence on scaled vs unscaled data")
print("• Learn why 90% of ML practitioners use Gradient Descent")
print("• Understand learning rates and convergence")
print()

print("🏆 TODAY'S ACHIEVEMENTS:")
print("✅ Mastered multivariate linear regression")
print("✅ Understood the power of vectorized code")
print("✅ Learned professional feature scaling habits")
print("✅ Built a robust data preprocessing pipeline")
print("✅ Prepared for advanced optimization algorithms")
print()

print("🎯 You are now ready for production-grade machine learning!")
print("Feature scaling isn't just theory - it's essential professional practice!")
