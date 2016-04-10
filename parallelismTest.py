from scoreSolutionGPU import knnLooGPU
from sklearn.neighbors import KNeighborsClassifier

from algorithms.utils import scoreSolution, loadDataSet, genInitSolution
import numpy as np

from algorithms.localSearch import bestFirst, bestFirstGPU
from algorithms.greedy import SFS, SFSGPU

import time

# Read data
data = loadDataSet("./data/wdbc.arff")
numSamples = data["features"].shape[0]
numFeatures = data["features"].shape[1]

# Init GPU score solution
scorerGPU = knnLooGPU(numSamples, numFeatures, 3)

# Initialize 3-NN classifier
knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

# CPU execution
start = time.time()
solutionCPU, scoreCPU = bestFirst(data["features"], data["target"],
                                  knnClassifier)
end = time.time()

timeCPU = end - start

# GPU execution
start = time.time()
solutionGPU, scoreGPU = bestFirstGPU(data["features"], data["target"],
                                     scorerGPU)
end = time.time()

timeGPU = end - start
if (solutionGPU == solutionCPU).all():
    print("Both solutions are the same. THE FUCKING SAME. C'MON!")
else:
    print("ERROR: Sorry, the solutions differ :(")

print("CPU:", scoreCPU, timeCPU)
print("GPU:", scoreGPU, timeGPU)
print("GPU execution is at least",
      int(np.floor(timeCPU / timeGPU)),
      "times faster.")


# # Set random seed
# np.random.seed(1)
# selectedFeatures = genInitSolution(numFeatures)
# selectedFeatures = np.ones(numFeatures, dtype=np.bool)
#
# # CPU execution
# start = time.time()
# scoreCPU = scoreSolution(data["features"][:, selectedFeatures],
#                          data["target"],
#                          knnClassifier)
# end = time.time()
#
# timeCPU = end - start
#
# # GPU execution
# start = time.time()
# scoreGPU = scorerGPU.scoreSolution(data["features"][:, selectedFeatures],
#                                    data["target"],
#                                    selectedFeatures.sum())
# end = time.time()
#
# timeGPU = end - start
# print("    ", "SCORE \t\t TIME")
# print("CPU:", scoreCPU, "\t", timeCPU)
# print("GPU:", scoreGPU, "\t", timeGPU)
# print("\nGPU execution is at least",
#       int(np.floor(timeCPU / timeGPU)),
#       "times faster.")
