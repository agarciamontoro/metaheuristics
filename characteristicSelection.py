import time
import numpy as np
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from algorithms.greedy import SFS
from algorithms.localSearch import bestFirst
from algorithms.simplePaths import simulatedAnnealing, tabuSearch


def loadDataSet(fileName):
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

    return data

if __name__ == "__main__":
    # Initialize 3-NN classifier
    knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

    # Define number of experiments
    numExperiments = 5

    # List of algorithms
    metaheuristics = [SFS, bestFirst, simulatedAnnealing, tabuSearch]

    # List of data sets
    datasets = ["./data/wdbc.arff",
                "./data/movement_libras.arff",
                "./data/arrhythmia.arff"]

    # Initialization of final data tables
    tables = {}
    for algorithm in metaheuristics:
        tables[algorithm.__name__] = np.zeros(shape=(10, 12))

    # Test all data sets
    for dataIdx, dataFileName in enumerate(datasets):
        data = loadDataSet(dataFileName)

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
                    algStr = algorithm.__name__

                    # Select features and measure time
                    start = time.time()
                    solution, scoreIn = algorithm(featuresTrain, targetTrain,
                                                  knnClassifier)
                    end = time.time()

                    # Fit the training data -run the classifier-
                    knnClassifier.fit(featuresTrain[:, solution],
                                      targetTrain)

                    # Get the score with the test data
                    scoreOut = knnClassifier.score(featuresTest[:, solution],
                                                   targetTest)

                    red = (len(solution) - solution.sum(0)) / len(solution)

                    print("ScoreIn  :\t", scoreIn)
                    print("ScoreOut :\t", scoreOut)
                    print("Reduction:\t", red)
                    print("Time     :\t", end-start)
                    print("*--"*50)

                    # Get results
                    results = [scoreIn, 100 * scoreOut, 100 * red, end - start]

                    # Populate table with data
                    init = 4*dataIdx
                    tables[algStr][2*exp + partIdx][init: init+4] = results

                    print(algStr, exp, partIdx, dataFileName)
                    print(tables[algStr])

                    now = str(int(time.time()))
                    np.savetxt("results/" + algStr + "_" + now + "_temp.csv",
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
        tables[algStr].append(meanValues)

        # Save the table :)
        np.savetxt("results/" + algStr + ".csv", tables[algStr],
                   delimiter=",",
                   fmt="%3.4f")
