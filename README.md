# metaheuristics
Learning metaheuristics through practice.

## Scorer
The solutions found by the algorithms to the characteristics selection problem are scored with the K nearest neighbours algorithm using the leave-one-out technique; i.e., for every sample `s`, find the K nearest neighbours, pick the most repeated class in those K elements and assign that label to `s`.

This scorer, which has a really high computational cost, has been implemented using PyCUDA, obtaining times 100x, 200x or even 300x times faster, depending on the number of samples of the dataset used.

The interface of the scorer is really simple, and if you want to use it you just have to follow these steps:

### Preliminars
* Install NumPy, PyCUDA and jinja2: this step depends on your operating system and distribution; e.g., in Arch Linux you should just type `pacaur -S python-{numpy,pycuda,jinja2}`. If you are used to `pip` you can use it, of course: `pip install numpy pycuda jinja2` :)
* Clone this repo to your computer or download the [interface](https://github.com/agarciamontoro/metaheuristics/blob/master/src/knnGPU/scoreSolutionGPU.py) and the [implementation](https://github.com/agarciamontoro/metaheuristics/blob/master/src/knnGPU/kernel.cu) of the scorer and put them in your project directory.

### Scorer construction

Call the scorer constructor passing it:

* The maximum number of samples of the data
* The maximum number of features of the data
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

* The `samples` array: a 2D numpy array, where the rows represent the samples and the columns the characteristics values. Note that you should pass the sliced data with only the selected characteristics.
* The `target` array: a 1D numpy array of length equal to the number of rows in `samples`. The i-th element of this array represents the class of the i-th sample in `samples`

For example:

```python
from knnLooGPU import knnLooGPU

# Data loading
data = loadDataSet(dataFileName)
numSamples = data["features"].shape[0]
numFeatures = data["features"].shape[1]

# Scorer constructor call
scorerGPU = knnLooGPU(numSamples, numFeatures, 3)

# Random characteristics selection solution
selectedFeatures = np.random.randint(2, size=numFeatures, dtype=np.bool)

# Get the score (from 0 to 100)
score = scorerGPU.scoreSolution(data[:, selectedFeatures], target)
```

And that's it!
