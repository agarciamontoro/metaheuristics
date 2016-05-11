import numpy as np
import numpy.ma as ma

def SFS(train, target, scorer):
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
            currentScore = scorer.scoreSolution(train[:, selectedFeatures],
                                                target)

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

def randomSFS(train, target, scorer):
    # Number of features in training data
    size = train.shape[1]

    # Define the solution and classification gains arrays
    selectedFeatures = np.zeros(size, np.bool)
    classifGains = np.zeros(size, np.float32)

    # Loop variables
    improvementFound = True
    bestScore = 0.

    # Model variables
    alpha = 0.3

    # Keep running until no improvement is found
    while improvementFound:
        # Loop variables
        improvementFound = False
        worstGain = 100
        bestGain = -100

        # Let's iterate through all not selected features
        notSelectedFeatures = np.where(selectedFeatures == False)[0]

        # For every feature not yet selected
        for feature in notSelectedFeatures:
            # Add the current feature to the solution
            selectedFeatures[feature] = True

            # Get the current score from the K-NN classifier
            currentScore = scorer(train[:, selectedFeatures], target)

            # Store the classification gain
            currentGain = currentScore - bestScore
            classifGains[feature] = currentGain

            # Update the best and worst gains
            if currentGain > bestGain:
                bestGain = currentGain
            if currentGain < worstGain:
                worstGain = currentGain

            # Return to the previous solution
            selectedFeatures[feature] = False

        # List of restricted candidates
        threshold = bestGain - alpha * (bestGain - worstGain)
        LRC = np.where(classifGains > threshold)[0]

        # Selection of best feature (randomly selected from LRC)
        bestFeature = np.random.choice(LRC, 1)

        bestGain = classifGains[bestFeature]

        if(bestGain > 0):
            selectedFeatures[bestFeature] = True
            bestScore = bestScore + bestGain
            improvementFound = True

            # Do not reconsider this vector entry for the threshold
            # computation
            classifGains[bestFeature] = -9999999999

    return selectedFeatures, bestScore
