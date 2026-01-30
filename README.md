# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries such as NumPy and Matplotlib.
2. Load the dataset containing city population and profit.
3. Initialize the parameters (slope and intercept) to zero.
4. Apply the gradient descent algorithm iteratively to update the parameters.
5. Calculate the cost function to reduce the prediction error.
6. Plot the training data and the regression line.
7. Display the final predicted output.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Ilakkiya K
RegisterNumber: 212225040130
*/
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Data
# -----------------------
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# -----------------------
# Parameters
# -----------------------
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)

losses = []

# -----------------------
# Gradient Descent
# -----------------------
for _ in range(epochs):
    y_hat = w * x + b

    # Mean Squared Error
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(12, 5))

# 1️⃣ Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# 2️⃣ Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```

## Output:
<img width="1253" height="587" alt="image" src="https://github.com/user-attachments/assets/3b549795-5b24-4c01-b63e-f9c4bff55cfb" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
