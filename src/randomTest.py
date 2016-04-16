import time
import numpy as np
import ntpath
import random
import os

from sklearn.neighbors import KNeighborsClassifier
from algorithms.utils import loadDataSet, getCPUScorer, genInitSolution
from knnGPU.knnLooGPU import knnLooGPU

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(19921201)
    random.seed(19921201)

    # Initialize CPU scorer
    knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)
    scorerCPU = getCPUScorer(knnClassifier)

    # List of data sets
    srcPath = os.path.dirname(os.path.realpath(__file__))
    basePath = os.path.join(srcPath, os.pardir)
    dataPath = os.path.join(basePath, "data")

    datasets = [os.path.join(dataPath, "wdbc.arff"),
                os.path.join(dataPath, "movement_libras.arff"),
                os.path.join(dataPath, "arrhythmia.arff")]

    numExperiments = 100

    # Errors
    errors = np.zeros(numExperiments, dtype=np.float32)

    # Test all data sets
    for dataIdx, dataFileName in enumerate(datasets):
        data = loadDataSet(dataFileName)

        numSamples = data["features"].shape[0]
        numFeatures = data["features"].shape[1]

        # Init GPU score solution
        scorerGPU = knnLooGPU(data["features"],
                              data["target"], 3).scoreSolution

        timeCPU = 0.
        timeGPU = 0.

        for i in range(numExperiments):
            selectedFeatures = genInitSolution(numFeatures)

            GPUmask = np.array(np.where(selectedFeatures == True)[0],
                               dtype=np.int32)

            # Select features and measure time with GPU
            start = time.time()
            scoreGPU = scorerGPU(GPUmask)
            end = time.time()

            timeGPU += end - start

            # Select features and measure time with CPU
            start = time.time()
            scoreCPU = scorerCPU(data["features"][:, selectedFeatures],
                                 data["target"])
            end = time.time()

            timeCPU += end - start

            errors[i] = scoreGPU - scoreCPU

        print("Database:   ", ntpath.basename(dataFileName))
        print("Time ratio  ", timeCPU/timeGPU)
        print("Error:      ", errors.mean())
        print()

    print("Done.")
