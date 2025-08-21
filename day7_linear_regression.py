import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

np.random.seed(42)

x = np.linspace(0, 10, 50)

noise = np.random.randn(50) * 2

y = 3 * x + 4 + noise

def hypothesis(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    return theta0 + theta1 * x

theta0 = 4
theta1 = 3

u_pred = hypothesis(x, theta0, theta1)

plt.scatter(x, y, label="Data (with noise)", color="blue")
plt.plot(x, u_pred, color="red", label="Hypothesis: h(x)={theta0}+{theta1}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Hypothesis function")
plt.legend()
plt.show()

#init point array with hardcoded values x,y
points = np.array([[2600, 550000], [3000, 565000], [3200, 610000], [3600, 680000], [4000, 725000]])

#load points into a pandas dataframe
df = pd.DataFrame(points, columns=["area", "Price"])

#plot df
plt.scatter(df["area"], df["Price"], color="green")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Prices")
# Format y-axis to show prices in thousands
plt.ticklabel_format(style='plain', axis='y')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
plt.show()

reg = linear_model.LinearRegression()
X = df[["area"]]
y = df["Price"]
reg.fit(X, y)

#predict area 3300
predicted_price = reg.predict([[3300]])
print(f"Predicted price for a house with 3300 sq ft area: ${predicted_price[0]:,.2f}")  
# Plotting the regression line
plt.scatter(df["area"], df["Price"], color="green", label="Data")
plt.plot(df["area"], reg.predict(X), color="red", label="Regression Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Prices with Regression Line")
# Format y-axis to show prices in thousands
plt.ticklabel_format(style='plain', axis='y')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
plt.legend()
plt.show()  

#cost function
def compute_cost(X: np.ndarray, y: np.ndarray, theta0: float, theta1: float) -> float:
    m = len(y)
    predictions = hypothesis(X, theta0, theta1)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost 
# Example usage of compute_cost
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
theta0 = 0
theta1 = 2
cost = compute_cost(X, y, theta0, theta1)
print(f"Cost for theta0={theta0}, theta1={theta1}: {cost:.2f}")

#Normal equation
def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best
# Example usage of normal_equation
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
theta_best = normal_equation(X, y)
print(f"Best theta from normal equation: {theta_best}")

plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, hypothesis(X, theta_best[0], theta_best[1]), color="red", label="Normal Equation Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Normal Equation Linear Regression")
plt.legend()
plt.show()

