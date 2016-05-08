import time
import numpy as np
from sklearn.cross_validation import StratifiedKFold

from algorithms.utils import loadDataSet

from algorithms.knn import knn
from algorithms.greedy import SFS
from algorithms.multiPaths import BMB, GRASP, ILS
from algorithms.genetic import stationaryGA, generationalGA

from knnGPU.knnLooGPU import knnLooGPU

import random
import subprocess
import os

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(19921201)
    random.seed(19921201)

    # Define number of experiments
    numExperiments = 5

    # List of algorithms
    metaheuristics = [generationalGA]

    # List of data sets
    srcPath = os.path.dirname(os.path.realpath(__file__))
    basePath = os.path.join(srcPath, os.pardir)
    dataPath = os.path.join(basePath, "data")
    resPath = os.path.join(basePath, "results", "03")
    tmpPath = os.path.join(resPath, "tmp")

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
                numTest = len(targetTest)

                # Init GPU score solution
                scorer = knnLooGPU(numSamples, numTest, numFeatures, 3)

                # Test all algorithms
                for algorithm in metaheuristics:
                    algStr = algorithm.__name__

                    scorer.resetCounter()

                    # Select features and measure time
                    start = time.time()
                    solution, scoreIn = algorithm(featuresTrain, targetTrain,
                                                  scorer)
                    end = time.time()

                    # Get the score with the test data
                    scoreOut = scorer.scoreOut(featuresTrain[:, solution],
                                               featuresTest[:, solution],
                                               targetTrain, targetTest)

                    # Compute reduction rate
                    red = (len(solution) - solution.sum(0)) / len(solution)

                    # Concatenate all the results
                    results = [scoreIn, scoreOut, 100 * red, end - start]

                    # Populate table with data
                    resSize = len(results)
                    init = resSize * dataIdx
                    tables[algStr][2*exp+partIdx][init:init+resSize] = results

                    # Save temp file
                    tempFileName = os.path.join(tmpPath, algStr + "_temp.csv")
                    np.savetxt(tempFileName,
                               tables[algStr],
                               delimiter=",",
                               fmt="%3.4f")

                partIdx += 1

    script = os.path.join(srcPath, "scripts", "finishTable.sh")

    for algorithm in metaheuristics:
        # Take algorithm name string
        algStr = algorithm.__name__

        # Compute mean values for every column
        meanValues = tables[algStr].mean(axis=0)

        # Append the mean values to the bottom of the table
        tables[algStr] = np.vstack((tables[algStr], meanValues))

        fileName = os.path.join(resPath, algStr + ".csv")

        # Save the table :)
        np.savetxt(fileName, tables[algStr], delimiter=",", fmt="%3.4f")

        # Prepend column and row names
        subprocess.call([script, fileName])
