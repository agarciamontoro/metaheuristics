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
    def __init__(self, numSamples, numTest, numFeatures, k):
        """ Constructor of the class.

        Arguments:
            numSamples: Number of the training samples in the input data.
            numTest: Number of the test samples in the input data.
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
        self.init(numSamples, numTest, numFeatures, k)

    def init(self, numSamples, numTest, numFeatures, k):
        """ Initialize (or reset) the environment of the scorer.

        Arguments:
            numSamples: Number of the samples in the input data.
            numFeatures: Number of features of each sample.
            k: Number of neighbours used to label the test sample.

        Returns nothing, but reset the scorer values and recompile the code
        against these new data.
        """

        # ==================== KERNEL TEMPLATE RENDERING ==================== #
        # Set the number of samples and features for the code compilation
        self.NUM_SAMPLES = numSamples
        self.NUM_TEST = numTest
        self.NUM_FEATURES = numFeatures

        # Compute the number of threads per block depending on the number of
        # samples and the hard-coded number of blocks.
        self.NUM_BLOCKS = 64
        self.NUM_THREADS_PER_BLOCK = np.ceil(self.NUM_SAMPLES /
                                             self.NUM_BLOCKS)

        # Input variables to renderize the template
        self.templateVars = {
            "MAX_NUM_SAMPLES": self.NUM_SAMPLES,
            "MAX_NUM_TEST": self.NUM_TEST,
            "MAX_NUM_FEATURES": self.NUM_FEATURES,
            "K": k
        }

        # Process the template to produce the final code.
        kernel = self.templateCode.render(self.templateVars)

        # ======================= KERNEL COMPILATION ======================= #

        # Compile the kernel code using pycuda.compiler
        self.compiledCode = compiler.SourceModule(kernel)

        # Get the kernel function from the compiled module
        self.GPUscoreSolution = self.compiledCode.get_function("scoreSolution")

        # Get the kernel function from the compiled module
        self.GPUscoreOut = self.compiledCode.get_function("scoreOut")

    def scoreSolution(self, samples, target):
        """ Computes the mean ratio of success using K-NN and leave-one-out

        For every sample in the samples numpy array:
            * Compute eucliden distance to all the **remaining** samples.
            * Extract the K nearest neighbours.
            * Label the sample using the most repeated class in the K nearest
            neighbours.
            * Check wether the predicted label is the actual one passed in the
            target numpy array

        Arguments:
            * samples: 2D numpy array, where the rows represent the samples
            and the columns the characteristics values.
            * target: 1D numpy array of length equal to the number of rows in
            samples. The i-th of this array represents the class of the i-th
            sample.

        Returns the mean ratio of success using K nearest neighbours as the
        target function and the leave-one-out technique.
        """
        # CPU binary array. The i-th value is 1 if the predicted label is
        # equal to the actual class and 0 if different.
        results = np.zeros(len(target), dtype=np.int32)

        # Number of samples and features. Necessary in the kernel code
        numSamples = samples.shape[0]
        numFeatures = samples.shape[1]

        # Transfer host (CPU) samples, target and results array to
        # device (GPU) memory
        samplesGPU = gpuarray.to_gpu(samples.flatten())
        targetGPU = gpuarray.to_gpu(target)
        resultsGPU = gpuarray.to_gpu(results)

        # Call the kernel on the card
        self.GPUscoreSolution(
            # Kernel function arguments
            samplesGPU,
            targetGPU,
            resultsGPU,
            np.int32(numFeatures),
            np.int32(numSamples),

            # CUDA memory configuration
            # Grid definition -> number of blocks x number of blocks.
            grid=(self.NUM_BLOCKS, 1, 1),
            # block definition -> number of threads x number of threads
            block=(int(self.NUM_THREADS_PER_BLOCK), 1, 1),
        )

        # Copy the results back from the device (GPU) memory to the host
        # (CPU) memory
        results = resultsGPU.get()

        # Compute the score, counting the number of success (1) and dividing
        # by the number of samples
        scoreGPU = sum(results)/len(results)

        # Returns the score from 0 to 100
        return 100*scoreGPU

    def scoreOut(self, samples, test, target_training, target_test):
        """ Computes the mean ratio of success in the test data using K-NN

        For every sample in the test numpy array:
            * Compute eucliden distance to all the samples.
            * Extract the K nearest neighbours.
            * Label the test sample using the most repeated class in the K nearest
            neighbours.
            * Check wether the predicted label is the actual one passed in the
            target numpy array

        Arguments:
            * samples: 2D numpy array, where the rows represent the samples
            and the columns the characteristics values.
            * test: 2D numpy array, where the rows represent the test samples
            and the columns the characteristics values.
            * target_training: 1D numpy array of length equal to the number of
            rows in samples. The i-th of this array represents the class of the
            i-th sample.
            * target_test: 1D numpy array of length equal to the number of rows
            in samples. The i-th of this array represents the class of the i-th
            sample.

        Returns the mean ratio of success using K nearest neighbours as the
        target function and the leave-one-out technique.
        """
        # Number of samples and features. Necessary in the kernel code
        numSamples = samples.shape[0]
        numTest = test.shape[0]
        numFeatures = samples.shape[1]

        # CPU binary array. The i-th value is 1 if the predicted label is
        # equal to the actual class and 0 if different.
        results = np.zeros(numTest, dtype=np.int32)

        # Transfer host (CPU) training and test samples, target and results
        # array to device (GPU) memory
        samplesGPU = gpuarray.to_gpu(samples.flatten())
        testGPU = gpuarray.to_gpu(test.flatten())
        target_trainingGPU = gpuarray.to_gpu(target_training)
        target_testGPU = gpuarray.to_gpu(target_test)
        resultsGPU = gpuarray.to_gpu(results)

        # Call the kernel on the card
        self.GPUscoreOut(
            # Kernel function arguments
            samplesGPU,
            testGPU,
            target_trainingGPU,
            target_testGPU,
            resultsGPU,
            np.int32(numFeatures),
            np.int32(numSamples),
            np.int32(numTest),

            # CUDA memory configuration
            # Grid definition -> number of blocks x number of blocks.
            grid=(self.NUM_BLOCKS, 1, 1),
            # block definition -> number of threads x number of threads
            block=(int(self.NUM_THREADS_PER_BLOCK), 1, 1),
        )

        # Copy the results back from the device (GPU) memory to the host
        # (CPU) memory
        results = resultsGPU.get()

        # Compute the score, counting the number of success (1) and dividing
        # by the number of samples
        scoreGPU = sum(results)/len(results)

        # Returns the score from 0 to 100
        return 100*scoreGPU
