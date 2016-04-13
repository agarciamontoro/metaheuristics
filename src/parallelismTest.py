import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold

from algorithms.utils import loadDataSet, getCPUScorer

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

    # Initialize CPU scorer
    knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
    scorerCPU = getCPUScorer(knnClassifier)

    # Define number of experiments
    numExperiments = 1

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
        data["features"] = data["features"][:100]
        data["target"] = data["target"][:100]

        numSamples = data["features"].shape[0]
        numFeatures = data["features"].shape[1]

        # Init GPU score solution
        scorerGPU = knnLooGPU(numSamples, numFeatures, 3)

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

                # Test all algorithms
                for algorithm in metaheuristics:
                    # Reset random seeds
                    np.random.seed(19921201)
                    random.seed(19921201)

                    # Select features and measure time with GPU
                    start = time.time()
                    solutionGPU, scoreInGPU = algorithm(featuresTrain,
                                                        targetTrain,
                                                        scorerGPU.scoreSolution
                                                        )
                    end = time.time()

                    timeGPU = end - start

                    # Reset random seeds
                    np.random.seed(19921201)
                    random.seed(19921201)

                    # Select features and measure time with CPU
                    start = time.time()
                    solutionCPU, scoreInCPU = algorithm(featuresTrain,
                                                        targetTrain,
                                                        scorerCPU)
                    end = time.time()

                    timeCPU = end - start

                    if (solutionGPU != solutionCPU).any() or\
                            scoreInCPU != scoreInGPU:

                        print("ERROR!   Database:  \t", dataFileName)
                        print("         Algorithm: \t", algorithm.__name__)
                        print("         ScoreInGPU:\t", scoreInGPU)
                        print("         ScoreInCPU:\t", scoreInCPU)
                    else:
                        print("Success! GPU:  ", timeGPU)
                        print("         CPU:  ", timeCPU)
                        print("         Ratio:", int(np.ceil(timeCPU/timeGPU)))

                partIdx += 1

    print("Done.")
