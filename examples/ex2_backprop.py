# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:13:07 2024

@author: Majdi Radaideh
"""

#------------------------------------------------------------------------------
#NERS 590: Applied Machine Learning for Nuclear Engineers
#In-class sript: Backpropagation demonstration for Feedforward NN 
#Date: 8/10/2024
#Author: Majdi I. Radaideh
#-----------------------------------------------------------------------------
# This is for a network with 2 inputs, one hidden layer with 2 neurons, and one output.

import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigma_0= 1 / (1 + np.exp(-x))
    return sigma_0 * (1-sigma_0)

# Training data
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

# Expected output for XOR logic
expected_output = np.array([[0],
                            [1],
                            [1],
                            [0]])

# Seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_layer_neurons = inputs.shape[1]   # Number of features in input data
hidden_layer_neurons = 2                # Number of hidden layer neurons
output_layer_neurons = 1                # Number of output layer neurons

# Weights and bias initialization
hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
output_bias = np.random.uniform(size=(1, output_layer_neurons))

# Learning rate
learning_rate = 0.1

# Training the network
for epoch in range(10000):
    # Forward Pass
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)
    
    # Calculate Error
    error = expected_output - predicted_output
    
    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        loss = np.mean(np.square(expected_output - predicted_output))
        print(f'Epoch {epoch + 1}, Loss: {loss}')

# Final predictions
print("Final predicted output:")
print(predicted_output)
print("Final expected output (not exactly macthing but certainly not bad for a very simple NN)")
print(expected_output)
