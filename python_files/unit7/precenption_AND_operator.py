import numpy as np

# Step 1: Define input and output for AND operator
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])  # AND output

# Step 2: Initialize weights and bias
weights = np.random.rand(2)
bias = np.random.rand(1)
learning_rate = 0.1

# Step 3: Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Step 4: Training the perceptron
epochs = 20
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        output = step_function(z)
        error = y[i] - output
        weights += learning_rate * error * X[i]
        bias += learning_rate * error
        print(f"Input: {X[i]}, Target: {y[i]}, Output: {output}, Error: {error}")
    print(f"Weights: {weights}, Bias: {bias}\n")

# Step 5: Testing
print("Final weights and bias:")
print(f"Weights: {weights}, Bias: {bias}")
print("\nTesting trained perceptron on AND inputs:")
for i in range(len(X)):
    z = np.dot(X[i], weights) + bias
    output = step_function(z)
    print(f"Input: {X[i]} => Output: {output}")
