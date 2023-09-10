#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets


# Activation functions and their derivatives
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d, lr):
        return d * (self.x > 0)


# Linear layer
class Affine:
    def __init__(self, n_input, n_output):
        self.w = np.random.normal(size=(n_input, n_output), scale=(2 / n_input) ** 0.5)
        self.b = np.zeros(n_output)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, d, lr):
        weight = self.w
        self.w -= lr * np.dot(self.x.T, d)
        self.b -= lr * np.sum(d, axis=0)
        return np.dot(d, weight.T)


# Loss function
class SoftmaxCrossEntropy:
    def forward(self, y, t):
        # Subtract max for numerical stability
        exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
        self.y = exp_y / np.sum(exp_y, axis=1, keepdims=True)
        self.t = t
        return -np.mean(np.sum(t * np.log(self.y), axis=1))

    def backward(self):
        return self.y - self.t


# Neural network model
class NeuralNetwork:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d, lr):
        for layer in reversed(self.layers):
            d = layer.backward(d, lr)


# Training function
def train(
    model, x_train, y_train, x_test, y_test, criterion, lr, n_steps, batch_size=16
):
    for step in range(1, n_steps + 1):
        idx = np.random.choice(len(x_train), batch_size)
        sample_x = x_train[idx]
        sample_y = y_train[idx]

        out = model.forward(sample_x)
        loss = criterion.forward(out, sample_y)
        d = criterion.backward()
        model.backward(d, lr)

        # Every 1000 steps, evaluate the model on the test data
        if step % 1000 == 0:
            predictions = model.forward(x_test).argmax(axis=1)
            accuracy = (predictions == y_test.argmax(axis=1)).mean()
            print(f"step: {step}, loss: {loss:.4f}, test accuracy: {accuracy:.4f}")


# Load dataset
digits = datasets.load_digits()
# Normalize data based on the max value (16 for 4-bit images)
x_train, x_test = digits.data[:1500] / 16.0, digits.data[1500:] / 16.0
y_train, y_test = np.eye(10)[digits.target[:1500]], np.eye(10)[digits.target[1500:]]

# Initialize and train the model
nn = NeuralNetwork(Affine(64, 24), ReLU(), Affine(24, 10))
train(nn, x_train, y_train, x_test, y_test, SoftmaxCrossEntropy(), 0.001, 10000, 16)
