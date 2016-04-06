import random
from collections import deque
from algorithms.utils import scoreSolution, genInitSolution, flip
import numpy as np

finalTemperature = 10**(-3)


def SA_getInitTemperature(mu, initialCost, phi):
    return (mu * initialCost) / -np.log(phi)


def SA_acceptWorseSolution(delta, temperature):
    randomValue = np.random.uniform(0., 1.)

    probability = np.exp(-delta / temperature)

    return randomValue <= probability


def SA_generateCoolingScheme(T0, TF, maxIter, size):
    maxGenerated = 10 * size
    maxAccepted = 0.1 * maxGenerated

    M = np.ceil(maxIter/maxGenerated)
    beta = (T0 - TF) / (M * T0 * TF)

    generatedNeighbourgs = 0

    def SA_cool(temperature):
        return temperature / (1 + beta * temperature)

    def SA_coolingNeeded(acceptedNeighbourgs):
        nonlocal generatedNeighbourgs
        generatedNeighbourgs += 1

        return (generatedNeighbourgs >= maxGenerated) or (acceptedNeighbourgs >= maxAccepted)

    return SA_cool, SA_coolingNeeded


def simulatedAnnealing(train, target, classifier):
    # Number of features in training data
    numFeatures = train.shape[1]

    selectedFeatures = genInitSolution(numFeatures)

    bestSolution = np.copy(selectedFeatures)

    bestScore = scoreSolution(train[:, bestSolution],
                              target,
                              classifier)

    currentScore = bestScore

    temperature = SA_getInitTemperature(0.3, bestScore, 0.3)

    maxIter = 15000

    SA_cool, SA_coolingNeeded = SA_generateCoolingScheme(temperature,
                                                         finalTemperature,
                                                         maxIter,
                                                         numFeatures)

    acceptedNeighbourgs = 1
    numEvaluations = 0

    while temperature > finalTemperature and acceptedNeighbourgs > 0 and numEvaluations < 15000:
        # Number of accepted neighbours
        acceptedNeighbourgs = 0

        while not SA_coolingNeeded(acceptedNeighbourgs):
            # Pick a random feature
            feature = np.random.randint(0, numFeatures)

            # Generate neighbour solution
            flip(selectedFeatures, feature)

            numEvaluations += 1

            # Get the current score from the K-NN classifier
            newScore = scoreSolution(train[:, selectedFeatures],
                                     target,
                                     classifier)

            delta = currentScore - newScore

            # Update current score
            if (delta < 0 or SA_acceptWorseSolution(delta, temperature)):
                currentScore = newScore
                acceptedNeighbourgs += 1

                # Update best score and scoreSolution
                if currentScore > bestScore:
                    bestScore = currentScore
                    bestSolution = np.copy(selectedFeatures)
            else:
                flip(selectedFeatures, feature)

        temperature = SA_cool(temperature)

    return bestSolution, bestScore


# =========================================================================== #
# =========================================================================== #


def tabuSearch(train, target, classifier):
    # Number of features in training data
    numSamples = train.shape[0]

    # Number of features in training data
    numFeatures = train.shape[1]

    # Initial and best solution
    selectedFeatures = genInitSolution(numFeatures)
    bestSolution = np.copy(selectedFeatures)

    # Initial and best score
    bestScore = scoreSolution(train[:, selectedFeatures],
                              target,
                              classifier)

    # Initialize tabu list with invalid indexes and fix its size to n/3
    tabuListDim = numFeatures // 3
    tabuList = deque([-1 for i in range(tabuListDim)], maxlen=tabuListDim)

    changedFeature = 0
    numEvaluations = 0

    while changedFeature is not None and numEvaluations < 15000:
        bestLocalScore = 0.
        changedFeature = None

        # For every solution in the neighbourhood
        for feature in random.sample(range(numFeatures), 30):
            # Generate neighbour solution
            flip(selectedFeatures, feature)

            # Get the current score from the K-NN classifier
            currentScore = scoreSolution(train[:, selectedFeatures],
                                         target,
                                         classifier)

            numEvaluations += 1

            # Reset to the local solution
            flip(selectedFeatures, feature)

            # Do not consider features in tabuList
            if(feature in tabuList):
                # Unless it produces a really good solution
                if(currentScore > bestScore):
                    bestScore = currentScore
                    bestSolution = np.copy(selectedFeatures)

            # Update best local neighbour
            elif(currentScore > bestLocalScore):
                bestLocalScore = currentScore
                changedFeature = feature

        # Add last change to tabu list and pick the new solution
        if(changedFeature is not None):
            tabuList.append(changedFeature)
            flip(selectedFeatures, changedFeature)

    return bestSolution, bestScore
