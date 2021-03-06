import numpy as np


def knn(train, target, scorer):
    # Number of features in training data
    size = train.shape[1]

    # Define the solution array
    selectedFeatures = np.ones(size, dtype=np.bool)

    # Get the current score from the K-NN classifier
    score = scorer(train[:, selectedFeatures], target)

    return selectedFeatures, score
