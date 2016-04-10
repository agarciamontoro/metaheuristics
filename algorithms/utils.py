import random
import numpy as np
from sklearn import cross_validation
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler


def loadDataSet(fileName):
    # Read data
    data, metaData = arff.loadarff(fileName)

    # Divide features data and classes classification
    train = data[metaData.names()[:-1]]
    target = data["class"]

    # Encapsulate all data in a dictionary
    data = {
        # Necessary to deal with numpy.void
        "features": np.asarray(train.tolist(), dtype=np.float32),
        "target": np.asarray(target.tolist(), dtype=np.int32)
    }

    # Normalize data
    normalizer = MinMaxScaler()
    data["features"] = normalizer.fit_transform(data["features"])

    return data



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
