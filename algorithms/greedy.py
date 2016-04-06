import numpy as np
from algorithms.utils import scoreSolution


def SFS(train, target, classifier):
    # Number of features in training data
    size = train.shape[1]

    # Define the solution array
    selectedFeatures = np.zeros(size, dtype=np.bool)

    # Loop variables
    improvementFound = True
    bestScore = 0.

    # Keep running until no improvement is found
    while improvementFound:
        # Loop variables
        improvementFound = False

        # Let's iterate through all not selected features
        notSelectedFeatures = np.where(selectedFeatures == False)[0]

        # For every feature not yet selected
        for feature in notSelectedFeatures:
            # Add the current feature to the solution
            selectedFeatures[feature] = True

            # Get the current score from the K-NN classifier
            currentScore = scoreSolution(train[:, selectedFeatures],
                                         target,
                                         classifier)

            # Update best score and solution
            if currentScore > bestScore:
                bestScore = currentScore
                bestFeature = feature
                improvementFound = True

            # Return to the previous solution if the score did not improve
            selectedFeatures[feature] = False

        if(improvementFound):
            selectedFeatures[bestFeature] = 1

    return selectedFeatures, bestScore
