import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import jinja2

# When importing this module we are initializing the device.
# Now, we can call the device and send information using
# the apropiate tools in the pycuda module.
import pycuda.autoinit

import os
basePath = os.path.dirname(__file__)


class knnLooGPU:
    """
    Leave-one-out scorer using K nearest neighbour algorithm as the target
    function for the characteristic selection problem.
    Implemented upon PyCUDA.
    """
    def __init__(self, samples, targets, k):
        """ Constructor of the class.

        Arguments:
            numSamples: Number of the samples in the input data.
            numFeatures: Number of features of each sample.
            k: Number of neighbours used to label the test sample.

        Returns the scorer with the environment ready to run over the input
        data.
        """
        # ==================== KERNEL TEMPLATE LOADING ==================== #

        # We must construct a FileSystemLoader object to load templates off
        # the filesystem
        templateLoader = jinja2.FileSystemLoader(searchpath=basePath)

        # An environment provides the data necessary to read and
        # parse our templates.  We pass in the loader object here.
        templateEnv = jinja2.Environment(loader=templateLoader)

        # Read the template file using the environment object.
        # This also constructs our Template object.
        self.templateCode = templateEnv.get_template("kernel.cu")

        # ========================= INITIALIZATION ========================= #
        self.init(samples, targets, k)

    def init(self, samples, targets, k):
        """ Initialize (or reset) the environment of the scorer.

        Arguments:
            numSamples: Number of the samples in the input data.
            numFeatures: Number of features of each sample.
            k: Number of neighbours used to label the test sample.

        Returns nothing, but reset the scorer values and recompile the code
        against these new data.
        """

        numSamples = samples.shape[0]
        numFeatures = samples.shape[1]
        # ==================== KERNEL TEMPLATE RENDERING ==================== #
        # Set the number of samples and features for the code compilation
        self.NUM_SAMPLES = numSamples
        self.NUM_FEATURES = numFeatures

        # Compute the number of threads per block depending on the number of
        # samples and the hard-coded number of blocks.
        self.NUM_BLOCKS = 64
        self.NUM_THREADS_PER_BLOCK = np.ceil(self.NUM_SAMPLES /
                                             self.NUM_BLOCKS)

        # Input variables to renderize the template
        self.templateVars = {
            "MAX_NUM_SAMPLES": self.NUM_SAMPLES,
            "MAX_NUM_FEATURES": self.NUM_FEATURES,
            "K": k
        }

        # Process the template to produce the final code.
        kernel = self.templateCode.render(self.templateVars)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        self.compiledCode = compiler.SourceModule(kernel)

        # CPU binary array. The i-th value is 1 if the predicted label is
        # equal to the actual class and 0 if different.
        results = np.zeros(numSamples, dtype=np.int32)

        # Load data into the GPU
        self.samplesGPU = gpuarray.to_gpu(samples.flatten())
        self.targetGPU = gpuarray.to_gpu(targets)
        self.resultsGPU = gpuarray.to_gpu(results)

        # Get the kernel function from the compiled module
        self.GPUscoreSolution = self.compiledCode.get_function("scoreSolution")

    def scoreSolution(self, idxMask):
        """ Computes the mean ratio of success using K-NN and leave-one-out

        For every sample in the samples numpy array:
            * Compute eucliden distance to all the **remaining** samples.
            * Extract the K nearest neighbours.
            * Label the sample using the most repeated class in the K nearest
            neighbours.
            * Check wether the predicted label is the actual one passed in the
            target numpy array

        Arguments:
            * mask: 1D numpy array of boolean, where false represent that
            the feature is not used and true that is used
            * samples: 2D numpy array, where the rows represent the samples
            and the columns the characteristics values.
            * target: 1D numpy array of length equal to the number of rows in
            samples. The i-th of this array represents the class of the i-th
            sample.

        Returns the mean ratio of success using K nearest neighbours as the
        target function and the leave-one-out technique.
        """

        # Number of samples and features. Necessary in the kernel code
        # numSamples = samples.shape[0]
        # numFeatures = samples.shape[1]

        # Transfer host (CPU) samples, target and results array to
        # device (GPU) memory

        idxMaskGPU = gpuarray.to_gpu(idxMask)

        # Call the kernel on the card
        self.GPUscoreSolution(
            # Kernel function arguments
            self.samplesGPU,
            self.targetGPU,
            self.resultsGPU,
            idxMaskGPU,
            np.int32(self.NUM_FEATURES),
            np.int32(self.NUM_SAMPLES),
            np.int32(idxMask.shape[0]),

            # CUDA memory configuration
            # Grid definition -> number of blocks x number of blocks.
            grid=(self.NUM_BLOCKS, 1, 1),
            # block definition -> number of threads x number of threads
            block=(int(self.NUM_THREADS_PER_BLOCK), 1, 1),
        )

        # Copy the results back from the device (GPU) memory to the host
        # (CPU) memory
        results = self.resultsGPU.get()

        # Compute the score, counting the number of success (1) and dividing
        # by the number of samples
        scoreGPU = sum(results)/len(results)

        # Returns the score from 0 to 100
        return 100*scoreGPU
