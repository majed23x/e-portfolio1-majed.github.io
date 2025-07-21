import numpy as np
import matplotlib.pyplot as plt

# === Activation Functions ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_out):
    return sigmoid_out * (1 - sigmoid_out)

# === Input and Output (XOR) ===
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

outputs = np.array([[0],
                    [1],
                    [1],
                    [0]])

# === Initialize Weights with Random Values ===
np.random.seed(42)  # Optional: reproducibility
weights_0 = 2 * np.random.random((2, 3)) - 1  # input to hidden
weights_1 = 2 * np.random.random((3, 1)) - 1  # hidden to output

# === Training Parameters ===
epochs = 400000
learning_rate = 0.6
error_log = []

# === Training Loop ===
for epoch in range(epochs):
    # Forward pass
    input_layer = inputs
    sum_synapse_0 = np.dot(input_layer, weights_0)
    hidden_layer = sigmoid(sum_synapse_0)

    sum_synapse_1 = np.dot(hidden_layer, weights_1)
    output_layer = sigmoid(sum_synapse_1)

    # Calculate error
    error_output_layer = outputs - output_layer
    avg_error = np.mean(np.abs(error_output_layer))

    if epoch % 100000 == 0:
        print(f"Epoch {epoch + 1}, Error: {avg_error}")
        error_log.append(avg_error)

    # Backpropagation
    derivative_output = sigmoid_derivative(output_layer)
    delta_output = error_output_layer * derivative_output

    weights_1_T = weights_1.T
    delta_output_weight = delta_output.dot(weights_1_T)
    delta_hidden_layer = delta_output_weight * sigmoid_derivative(hidden_layer)

    # Update weights
    hidden_layer_T = hidden_layer.T
    weights_1 += hidden_layer_T.dot(delta_output) * learning_rate

    input_layer_T = input_layer.T
    weights_0 += input_layer_T.dot(delta_hidden_layer) * learning_rate

# === Final Evaluation ===
print("\n=== Final Weights ===")
print("Weights from input to hidden:\n", weights_0)
print("Weights from hidden to output:\n", weights_1)

print("\n=== Final Predictions ===")
print("Expected outputs:\n", outputs)
print("Predicted outputs:\n", np.round(output_layer, 3))

# === Plotting Error ===
plt.plot(range(0, epochs, 100000), error_log)
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training Error over Time")
plt.grid(True)
plt.show()

# === Predict Function ===
def calculate_output(instance):
    hidden = sigmoid(np.dot(instance, weights_0))
    output = sigmoid(np.dot(hidden, weights_1))
    return output[0]

# === Test Predictions ===
print("\n=== Test XOR Inputs ===")
for x in inputs:
    pred = round(calculate_output(x))
    print(f"Input: {x} => Predicted: {pred}")
