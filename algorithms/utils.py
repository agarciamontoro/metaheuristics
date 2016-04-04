import numpy as np
from sklearn import cross_validation


def scoreSolution(train, target, knnClassifier):
    size = train.shape[1]

    leaveOneOut = cross_validation.LeaveOneOut(size)

    finalScore = 0.

    for trainIdx, testIdx in leaveOneOut:
        looDataTrain = train[trainIdx]
        looDataTest = train[testIdx]

        looTargetTrain = target[trainIdx]
        looTargetTest = target[testIdx]

        # Test it with knnClassifier
        knnClassifier.fit(looDataTrain, looTargetTrain)
        finalScore += knnClassifier.score(looDataTest, looTargetTest)

    return finalScore / size


# Returns the initial solution
def genInitSolution(solSize):
    return np.random.randint(2, size=solSize)
