import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold

from algorithms.utils import loadDataSet

from algorithms.knn import knn
from algorithms.greedy import SFS
from algorithms.localSearch import bestFirst
from algorithms.simplePaths import simulatedAnnealing, tabuSearch

from knnGPU.knnLooGPU import knnLooGPU

import random

import os

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(19921201)
    random.seed(19921201)

    # Initialize 3-NN classifier
    knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

    # Define number of experiments
    numExperiments = 5

    # List of algorithms
    metaheuristics = [knn, SFS, bestFirst, simulatedAnnealing, tabuSearch]

    # List of data sets
    srcPath = os.path.dirname(os.path.realpath(__file__))
    basePath = os.path.join(srcPath, os.pardir)
    dataPath = os.path.join(basePath, "data")

    datasets = [os.path.join(dataPath, "wdbc.arff"),
                os.path.join(dataPath, "movement_libras.arff"),
                os.path.join(dataPath, "arrhythmia.arff")]

    # Initialization of final data tables
    tables = {}
    for algorithm in metaheuristics:
        tables[algorithm.__name__] = np.zeros(shape=(10, 12))

    # Test all data sets
    for dataIdx, dataFileName in enumerate(datasets):
        data = loadDataSet(dataFileName)

        # Define number of features
        numFeatures = data["features"].shape[1]

        # Repeat the experiment numExperiments times
        for exp in range(numExperiments):
            # Make the partitions
            partitions = StratifiedKFold(data["target"], 2, shuffle=True)
            partIdx = 0

            for idxTrain, idxTest in partitions:
                # Define training data partitions
                featuresTrain = data["features"][idxTrain]
                targetTrain = data["target"][idxTrain]

                # Define test data partitions
                featuresTest = data["features"][idxTest]
                targetTest = data["target"][idxTest]

                # Define number of train and test samples
                numSamples = len(targetTrain)
                numSamples = len(targetTest)

                # Init GPU score solution
                scorerGPU = knnLooGPU(numSamples, numTest, numFeatures, 3)

                # Test all algorithms
                for algorithm in metaheuristics:
                    algStr = algorithm.__name__

                    # Select features and measure time
                    start = time.time()
                    solution, scoreIn = algorithm(featuresTrain, targetTrain,
                                                  scorerGPU.scoreSolution)
                    end = time.time()

                    # Fit the training data -run the classifier-
                    knnClassifier.fit(featuresTrain[:, solution],
                                      targetTrain)

                    # Get the score with the test data
                    scoreOut = scoreOut(featuresTrain[:, solution], featuresTest[:, solution],
                                        targetTrain, targetTest)

                    red = (len(solution) - solution.sum(0)) / len(solution)

                    # Get results
                    results = [scoreIn, 100 * scoreOut, 100 * red, end - start]

                    # Populate table with data
                    init = 4*dataIdx
                    tables[algStr][2*exp + partIdx][init: init+4] = results

                    # Save temp file
                    np.savetxt("results/" + algStr + "_temp.csv",
                               tables[algStr],
                               delimiter=",",
                               fmt="%3.4f")

                partIdx += 1

    for algorithm in metaheuristics:
        # Take algorithm name string
        algStr = algorithm.__name__

        # Compute mean values for every column
        meanValues = tables[algStr].mean(axis=0)

        # Append the mean values to the bottom of the table
        tables[algStr] = np.vstack((tables[algStr], meanValues))

        # Save the table :)
        np.savetxt("results/" + algStr + ".csv", tables[algStr],
                   delimiter=",",
                   fmt="%3.4f")
