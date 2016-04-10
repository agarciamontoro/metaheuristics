from algorithms.utils import loadDataSet, scoreSolution
from sklearn.neighbors import KNeighborsClassifier

import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

# Read data
data = loadDataSet("./data/wdbc.arff")

NUM_SAMPLES = data["features"].shape[0]
NUM_FEATURES = data["features"].shape[1]

NUM_BLOCKS = 64
NUM_THREADS_PER_BLOCK = np.ceil(NUM_SAMPLES / NUM_BLOCKS)

print(NUM_BLOCKS, NUM_THREADS_PER_BLOCK)

# Initialize 3-NN classifier
knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

scoreSolution(data["features"], data["target"], knnClassifier)

# ======================== KERNEL TEMPLATE RENDERING ======================== #

# We must construct a FileSystemLoader object to load templates off
# the filesystem
templateLoader = jinja2.FileSystemLoader(searchpath="./")

# An environment provides the data necessary to read and
# parse our templates.  We pass in the loader object here.
templateEnv = jinja2.Environment(loader=templateLoader)

# Read the template file using the environment object.
# This also constructs our Template object.
template = templateEnv.get_template("kernel.cu")

# Specify any input variables to the template as a dictionary.
templateVars = {
    "NUM_SAMPLES": NUM_SAMPLES,
    "NUM_FEATURES": NUM_FEATURES,
    "K": 3
}

# Finally, process the template to produce our final text.
kernel = template.render(templateVars)

# =========================== KERNEL COMPILATION =========================== #

# Compile the kernel code using pycuda.compiler
mod = compiler.SourceModule(kernel)

# Get the kernel function from the compiled module
GPUscoreSolution = mod.get_function("scoreSolution")

# ============================ ACTUAL EXECUTION ============================ #

results = np.zeros(len(data["target"]), dtype=np.int32)

# Transfer host (CPU) memory to device (GPU) memory
featuresGPU = gpuarray.to_gpu(data["features"])
targetGPU = gpuarray.to_gpu(data["target"])
resultsGPU = gpuarray.to_gpu(results)

# Create two timers for measuring time
start = driver.Event()
end = driver.Event()

start.record()  # start timing

# Call the kernel on the card
GPUscoreSolution(
    # inputs
    featuresGPU,
    targetGPU,
    resultsGPU,

    # Grid definition -> number of blocks x number of blocks.
    grid=(NUM_BLOCKS, 1, 1),
    # block definition -> number of threads x number of threads
    block=(int(NUM_THREADS_PER_BLOCK), 1, 1),
)

results = resultsGPU.get()
scoreGPU = sum(results)/len(results)

end.record()

end.synchronize()
timeGPU = start.time_till(end)*1e-3

# Initialize 3-NN classifier
knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

start = time.time()
scoreCPU = scoreSolution(data["features"], data["target"], knnClassifier)
end = time.time()

timeCPU = end - start

print("CPU:", scoreCPU, timeCPU)
print("GPU:", 100*scoreGPU, timeGPU)
print("La ejecución en CPU es al menos",
      int(np.floor(timeCPU / timeGPU)),
      "veces más rápido.")
