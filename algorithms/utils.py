import random
import numpy as np
from sklearn import cross_validation


def scoreSolution(data, target, knnClassifier):
    # Number of samples in data training
    size = data.shape[0]

    leaveOneOut = cross_validation.LeaveOneOut(size)

    finalScore = 0.

    for trainIdx, testIdx in leaveOneOut:
        looDataTrain = data[trainIdx]
        looDataTest = data[testIdx]

        looTargetTrain = target[trainIdx]
        looTargetTest = target[testIdx]

        # Test it with knnClassifier
        knnClassifier.fit(looDataTrain, looTargetTrain)
        finalScore += knnClassifier.score(looDataTest, looTargetTest)

    return 100 * (finalScore / size)


# Returns the initial solution
def genInitSolution(solSize):
    solution = np.random.randint(2, size=solSize)
    return np.asarray(solution, dtype=np.bool)


def flip(solution, feature):
    solution[feature] = not solution[feature]


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)
