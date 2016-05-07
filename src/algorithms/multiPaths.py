from algorithms.utils import genInitSolution
from algorithms.localSearch import bestFirst
from algorithms.greedy import randomSFS

import numpy as np


def BMB(train, target, scorer):

    bestSolution, bestScore = None, -1

    for i in range(25):
        currentSolution, currentScore = bestFirst(train, target, scorer)

        if currentScore > bestScore:
            bestSolution = np.copy(currentSolution)
            bestScore = currentScore

    return bestSolution, bestScore


def GRASP(train, target, scorer):
    # Define initial solution and score
    bestSolution, bestScore = None, 0

    # Repeat it 25 times
    for i in range(25):
        # Generate greedy random solution
        currentSolution, currentScore = randomSFS(train, target, scorer)

        # Improve previous solution through local search
        currentSolution, currentScore = bestFirst(train, target, scorer,
                                                  currentSolution)

        # Update best solution and score
        if currentScore > bestScore:
            bestScore = currentScore
            bestSolution = np.copy(currentSolution)

    return bestSolution, bestScore


def mutateSolution(initSol, perc=0.1):
    # Number of features in training data
    size = len(initSol)

    mutatedIndices = np.random.randint(0, size, int(perc*size))

    mutatedSol = np.copy(initSol)
    mutatedSol[mutatedIndices] = (initSol[mutatedIndices] + 1) % 2

    return mutatedSol


def ILS(train, target, scorer):
    # Number of features in training data
    size = train.shape[1]

    initSolution = genInitSolution(size)
    bestSolution, bestScore = bestFirst(train, target, scorer, initSolution)

    prevSolution, prevScore = bestSolution, bestScore

    for i in range(24):
        mutation = mutateSolution(prevSolution)
        currentSolution, currentScore = bestFirst(train, target, scorer,
                                                  mutation)

        # Acceptance criteria
        if currentScore > prevScore:
            prevSolution = np.copy(currentSolution)
            prevScore = currentScore

        # Best solution update
        if prevScore > bestScore:
            bestScore = prevScore
            bestSolution = prevSolution

    return bestSolution, bestScore
