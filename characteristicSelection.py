import time
import numpy as np
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from algorithms.greedy import SFS
from algorithms.localSearch import bestFirst
from algorithms.simplePaths import simulatedAnnealing, tabuSearch

# Read data
data, metaData = arff.loadarff("./data/wdbc.arff")

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

# Initialize 3-NN classifier
knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

# Define number of experiments
numExperiments = 5

# List of algorithms
metaheuristics = [SFS, bestFirst, simulatedAnnealing, tabuSearch]

for i in range(numExperiments):
    # Make the partitions
    partitions = StratifiedKFold(data["target"], 2, shuffle=True)

    for idxTrain, idxTest in partitions:
        # Define training data partitions
        featuresTrain = data["features"][idxTrain]
        targetTrain = data["target"][idxTrain]

        # Define test data partitions
        featuresTest = data["features"][idxTest]
        targetTest = data["target"][idxTest]

        # Select features and measure time
        start = time.time()
        solution = tabuSearch(featuresTrain, targetTrain, knnClassifier)
        end = time.time()

        print(solution)

        # Fit the training data -run the classifier-
        knnClassifier.fit(featuresTrain[:, solution], targetTrain)

        # Get the score with the test data
        score = knnClassifier.score(featuresTest[:, solution],
                                    targetTest)

        red = (len(solution) - solution.sum(0)) / len(solution)

        print("Tiempo   :\t", end-start)
        print("Score    :\t", score)
        print("Reduction:\t", red)
        print("*--"*50, "*")
