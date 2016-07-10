from genetic import generationalPopulation


def AM(train, target, scorer, memeticModel):
    def genScorer(chromosome):
        return scorer.scoreSolution(train[:, chromosome], target)

    # Number of features
    size = train.shape[1]

    population = generationalPopulation(size, genScorer, populationSize=10,
                                        crossoverProb=0.7, mutationProb=0.001,
                                        hybridModel=memeticModel)

    # Evolution
    while(scorer.scoreCalls < 15000):
        population.selection()
        population.recombination()
        population.mutation()
        population.replacement()
        population.localImprovement()

    bestChromosome = population.bestChromosome()

    return bestChromosome.genes, bestChromosome.score


def AM1010(train, target, scorer):
    return AM(train, target, scorer, "1010")


def AM1001(train, target, scorer):
    return AM(train, target, scorer, "1001")


def AM1001M(train, target, scorer):
    return AM(train, target, scorer, "1001M")
