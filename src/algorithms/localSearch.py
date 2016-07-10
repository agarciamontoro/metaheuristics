from algorithms.utils import genInitSolution, flip, randomly


def bestFirst(train, target, scorer, initSol=None):
    # Number of features in training data
    size = train.shape[1]

    if initSol is not None:
        selectedFeatures = initSol
    else:
        selectedFeatures = genInitSolution(size)

    improvementFound = True
    bestScore = scorer(train[:, selectedFeatures], target)

    iterations = 0

    while improvementFound or iterations > 15000:
        improvementFound = False

        for feature in randomly(range(size)):
            flip(selectedFeatures, feature)

            # Get the current score from the K-NN classifier
            currentScore = scorer(train[:, selectedFeatures], target)

            iterations += 1

            # Update best score and solution
            if currentScore > bestScore:
                bestScore = currentScore
                improvementFound = True
                break
            else:
                flip(selectedFeatures, feature)

    return selectedFeatures, bestScore
