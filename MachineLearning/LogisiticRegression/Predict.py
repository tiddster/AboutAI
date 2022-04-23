import Sigmoid
import numpy as np


def predict(theta, X):
    probabilities = Sigmoid.sigmoid(X @ theta)
    probabilities = np.array(probabilities)
    return [0 if p < 0.5 else 1 for p in probabilities.flatten()]


def getAccuracy(theta, X, y):
    predictions = predict(theta, X)
    total = len(predictions)
    right = 0
    for a, b in zip(predictions, y):
        right += 1 if a == b else 0
    return right / total