from algorithms.utils import scoreSolution, genInitSolution, flip, randomly


def bestFirst(train, target, classifier):
    # Number of features in training data
    size = train.shape[1]

    selectedFeatures = genInitSolution(size)

    improvementFound = True
    bestScore = scoreSolution(train[:, selectedFeatures],
                              target,
                              classifier)

    while improvementFound:
        improvementFound = False

        for feature in randomly(range(size)):
            flip(selectedFeatures, feature)

            # Get the current score from the K-NN classifier
            currentScore = scoreSolution(train[:, selectedFeatures],
                                         target,
                                         classifier)

            # Update best score and solution
            if currentScore > bestScore:
                bestScore = currentScore
                improvementFound = True
                break
            else:
                flip(selectedFeatures, feature)

    return selectedFeatures, bestScore
