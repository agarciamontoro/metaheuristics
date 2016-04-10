import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

class knnLooGPU:
    def __init__(self, numSamples, numFeatures, k):
        self.NUM_SAMPLES = numSamples
        self.NUM_FEATURES = numFeatures

        self.NUM_BLOCKS = 64
        self.NUM_THREADS_PER_BLOCK = np.ceil(self.NUM_SAMPLES /
                                             self.NUM_BLOCKS)

        # ==================== KERNEL TEMPLATE RENDERING ==================== #

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
            "NUM_SAMPLES": self.NUM_SAMPLES,
            "NUM_FEATURES": self.NUM_FEATURES,
            "K": k
        }

        # Finally, process the template to produce our final text.
        kernel = template.render(templateVars)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        mod = compiler.SourceModule(kernel)

        # Get the kernel function from the compiled module
        self.GPUscoreSolution = mod.get_function("scoreSolution")

    def scoreSolution(self, features, target):
        results = np.zeros(len(target), dtype=np.int32)

        # Transfer host (CPU) memory to device (GPU) memory
        featuresGPU = gpuarray.to_gpu(features)
        targetGPU = gpuarray.to_gpu(target)
        resultsGPU = gpuarray.to_gpu(results)

        # Create two timers for measuring time
        start = driver.Event()
        end = driver.Event()

        start.record()  # start timing

        # Call the kernel on the card
        self.GPUscoreSolution(
            # inputs
            featuresGPU,
            targetGPU,
            resultsGPU,

            # Grid definition -> number of blocks x number of blocks.
            grid=(self.NUM_BLOCKS, 1, 1),
            # block definition -> number of threads x number of threads
            block=(int(self.NUM_THREADS_PER_BLOCK), 1, 1),
        )

        results = resultsGPU.get()
        scoreGPU = sum(results)/len(results)

        end.record()
        end.synchronize()

        timeGPU = start.time_till(end)*1e-3

        return scoreGPU, timeGPU
