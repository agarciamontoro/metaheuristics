from algorithms.localSearch import bestFirst

def BMB(train, target, scorer):

    bestSolution, bestScore = None, -1

    for i in range(25):
        currentSolution, currentScore = bestFirst(train, target, scorer)

        if currentScore > bestScore:
            bestSolution = currentSolution
            bestScore = currentScore

    return bestSolution, bestScore

# def GRASP(train, target, scorer):
#
