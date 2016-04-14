# metaheuristics
Learning metaheuristics through practice.

## Massively parallel leave-one-out kNN scorer. Why?

The solutions found by the metaheuristics implemented in this project to solve the characteristics selection problem are scored with the K nearest neighbours algorithm using the leave-one-out technique; i.e., for every sample `s`, find the K nearest neighbours, pick the most repeated class in those K elements and assign that label to `s`.

This scorer, which has a really high computational cost, has been implemented using PyCUDA, obtaining times 100x, 200x or even 300x times faster, depending on the number of samples of the dataset used.

## How to use the parallelized scorer?

### TL;DR

Install numpy, pycuda and jinja2 and copy the [interface](https://github.com/agarciamontoro/metaheuristics/blob/master/src/knnGPU/knnLooGPU.py) and the [implementation](https://github.com/agarciamontoro/metaheuristics/blob/master/src/knnGPU/kernel.cu) of the scorer to your project. Build an object of the knnLooGPU class with the needed number of samples and features and the *k* of the *k*-NN. Call the scorerSolution method of the returned object with the samples sliced to the selected features and the target. Done!

===========
### Detailed instructions

#### Preliminars
* Install NumPy, PyCUDA and jinja2: this step depends on your operating system and distribution; e.g., in Arch Linux you should just type `pacaur -S python-{numpy,pycuda,jinja}`. If you are used to `pip` you can alternatively do the following: `pip install numpy pycuda jinja2` :)
* Clone this repo to your computer or just download the [interface](https://github.com/agarciamontoro/metaheuristics/blob/master/src/knnGPU/knnLooGPU.py) and the [implementation](https://github.com/agarciamontoro/metaheuristics/blob/master/src/knnGPU/kernel.cu) of the scorer and put them in your project directory.

### Scorer construction

Call the scorer constructor passing it:

* The maximum number of samples of the data.
* The maximum number of features of the data.
* The number of nearest neighbours considered. Yep, just the **K** in the **K**-NN

For example:

```python
from knnLooGPU import knnLooGPU

# Data loading
data = loadDataSet(dataFileName)
numSamples = data["features"].shape[0]
numFeatures = data["features"].shape[1]

# Scorer constructor call
scorerGPU = knnLooGPU(numSamples, numFeatures, 3)
```

### Scorer call

Call the scorer using the `scoreSolution` method of the `scorerGPU` object, passing it:

* The `samples` array: a 2D numpy array with type `dtype = np.float32`, where the rows represent the samples and the columns the characteristics values of each sample. Note that you should pass the already sliced data with only the selected characteristics.
* The `target` array: a 1D numpy array with type `dtype = np.int32` and length equal to the number of rows in `samples`. The i-th element of this array represents the class of the i-th sample in `samples`

We can now complete our previous example:

```python
from knnLooGPU import knnLooGPU

# Data loading
data = loadDataSet(dataFileName)
numSamples = data["features"].shape[0]
numFeatures = data["features"].shape[1]

# Scorer constructor call
scorerGPU = knnLooGPU(numSamples, numFeatures, 3)

# This is a random solution
selectedFeatures = np.random.randint(2, size=numFeatures, dtype=np.bool)

# Slice the 2D data array to have just the desired columns
sclideData = data[:, selectedFeatures]

# Get the score! The result has to be interpreted as a success percentage (from 0 to 100)
score = scorerGPU.scoreSolution(, target)
```
And that's it!

#### Final considerations
Please, note that the arrays follow a data type convention quite strict. This is **very important**:
* The data with the samples has to be a numpy 2D array with `dtype = np.float32`.
* The target array with the classes has to be a numpy 1D array with `dtype= np.int32`.

Furthermore, take into account that every time you call the `knnLooGPU` class constructor, the C code file where the magic happens has to be compiled. Don't worry, this is done automatically by PyCUDA, but it will take some time, as all compilation processes do. Then, you should call the constructor **just once** for every database ---don't do it every time you call an algorithm, or every time you change your data partition---, with the **maximum** number of samples and the **maximum** number of characteristics the database have.
