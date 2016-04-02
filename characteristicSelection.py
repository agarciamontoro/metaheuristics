import numpy as np
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold

# Read data
data, metaData = arff.loadarff("./data/arrhythmia.arff")

# Divide features data and classes classification
train = data[:][metaData.names()[:-1]]
target = data[:]["class"]

# Encapsulate all data in a dictionary
data = {
    # Necessary to deal with numpy.void
    "train": np.asarray(train.tolist(), dtype=np.float32),
    "target": target
}

skf = StratifiedKFold(data["target"], 2, shuffle=True)

for idxTrain, idxTest in skf:
    # Initialize 3-NN classifier
    knnClassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=8)

    # Fit the data -run the classifier-
    knnClassifier.fit(data["train"][idxTrain], data["target"][idxTrain])

    # Get the score
    score = knnClassifier.score(data["train"][idxTest],
                                data["target"][idxTest])
                                
    print(score)
