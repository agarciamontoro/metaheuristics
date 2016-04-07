import numpy as np
from algorithms.utils import scoreSolution


def knn(train, target, classifier):
    # Number of features in training data
    size = train.shape[1]

    # Define the solution array
    selectedFeatures = np.ones(size, dtype=np.bool)

    # Get the current score from the K-NN classifier
    score = scoreSolution(train[:, selectedFeatures], target, classifier)

    return selectedFeatures, score
