from scoreSolutionGPU import knnLooGPU
from sklearn.neighbors import KNeighborsClassifier

from algorithms.utils import scoreSolution, loadDataSet
import numpy as np

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
scoreCPU = scoreSolution(data["features"], data["target"], knnClassifier)
end = time.time()

timeCPU = end - start

# GPU execution
start = time.time()
scoreGPU, _ = scorerGPU.scoreSolution(data["features"], data["target"])
end = time.time()

timeGPU = end - start
print("CPU:", scoreCPU, timeCPU)
print("GPU:", 100*scoreGPU, timeGPU)
print("GPU execution is at least",
      int(np.floor(timeCPU / timeGPU)),
      "times faster.")
