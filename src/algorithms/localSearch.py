from algorithms.utils import genInitSolution, flip, randomly


def bestFirst(train, target, scorer):
    # Number of features in training data
    size = train.shape[1]

    selectedFeatures = genInitSolution(size)

    improvementFound = True
    bestScore = scorer(train[:, selectedFeatures], target)

    while improvementFound:
        improvementFound = False

        for feature in randomly(range(size)):
            flip(selectedFeatures, feature)

            # Get the current score from the K-NN classifier
            currentScore = scorer(train[:, selectedFeatures], target)

            # Update best score and solution
            if currentScore > bestScore:
                bestScore = currentScore
                improvementFound = True
                break
            else:
                flip(selectedFeatures, feature)

    return selectedFeatures, bestScore
