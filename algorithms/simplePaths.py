from algorithms.utils import scoreSolution, genInitSolution, flip, randomly
import math
import numpy as np

finalTemperature = 0.001

def SA_getInitTemperature(mu, initialCost, phi):
    return (mu * initialCost) / -math.log(phi)


def SA_neighborhood(temperature):
    pass


def SA_acceptWorseSolution(delta, temperature):
    randomValue = np.random.uniform(0., 1.)
    probability = math.exp(-delta / temperature)

    return randomValue <= probability


def SA_generateCoolingScheme(T0, TF, M):
    beta = (T0 - TF) / (M * T0 * TF)

    def cool(temperature):
        return temperature / (1 + beta * temperature)

    return cool


def simulatedAnnealing(train, target, classifier):
    # Number of features in training data
    size = train.shape[1]

    selectedFeatures = genInitSolution(size)

    bestSolution = np.copy(selectedFeatures)

    bestScore = scoreSolution(train[:, selectedFeatures],
                              target,
                              classifier)

    temperature = SA_getInitTemperature(0.3, bestScore, 0.3)

    SA_cool = SA_generateCoolingScheme(temperature, finalTemperature, 15000)

    while temperature < finalTemperature:
        for feature in SA_neighborhood(temperature):
            # Generate new solution
            flip(selectedFeatures, feature)

            # Get the current score from the K-NN classifier
            currentScore = scoreSolution(train[:, selectedFeatures],
                                         target,
                                         classifier)

            delta = currentScore - bestScore

            # Update best score and solution
            if delta > 0:
                bestScore = currentScore
                bestSolution = np.copy(selectedFeatures)
            elif SA_acceptWorseSolution(delta, temperature):
                flip(selectedFeatures, feature)

        temperature = SA_cool(temperature, M)

    return selectedFeatures
