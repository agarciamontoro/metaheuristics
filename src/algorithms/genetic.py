from algorithms.utils import flip, genInitSolution, pairwise, randomly

from operator import attrgetter
import numpy as np
from copy import deepcopy
from itertools import chain


class Chromosome:
    def __init__(self, chromosomSize, scorer, HUX=False):
        self.size = chromosomSize
        self.genes = genInitSolution(self.size)
        self.HUX = HUX

        self.scorer = scorer
        self.setScore()
        self.crossover = self.HUXcrossover if HUX else self.TRIcrossover

    def setScore(self):
        self.score = self.scorer(self.genes)

    def mutateGene(self, gene=None):
        if gene is None:
            gene = np.random.randint(self.size)

        # Mutation!
        flip(self.genes, gene)

        # Update scoring
        self.setScore()

    def TRIcrossover(self, partner):
        # Generate the crossover points
        pts = np.random.choice(self.size, 2)

        # Generate the new childs from their parents
        child1 = deepcopy(self)
        child2 = deepcopy(partner)

        # Do the crossover!
        child1.genes[pts[0]:pts[1]] = partner.genes[pts[0]:pts[1]]
        child2.genes[pts[0]:pts[1]] = self.genes[pts[0]:pts[1]]

        # Update the scores
        child1.setScore()
        child2.setScore()

        return [child1, child2]

    def HUXcrossover(self, partner):
        # Generate the new childs from their parents
        child1 = deepcopy(self)
        child2 = deepcopy(partner)

        difFeatures = np.where(child1 != child2)

        for child in [child1, child2]:
            for f in difFeatures:
                if(np.random.uniform(0., 1.) > 0.5):
                    flip(child.genes, f)

        # Update the scores
        child1.setScore()
        child2.setScore()

        return [child1, child2]

    def localImprovement(self):
        # Initial score
        bestScore = self.score

        for gene in randomly(range(self.size)):
            flip(self.genes, gene)

            # Get the current score from the K-NN classifier
            currentScore = self.scorer(self.genes)

            # If the current solution is better, update the bestScore,
            # keep the changes and finish. If it is not, undo the gene flip
            # and continue.
            if currentScore > bestScore:
                bestScore = currentScore
                break
            else:
                flip(self.genes, gene)

        self.score = bestScore


# General
class Population:
    def __init__(self, populationSize, chromosomeSize, scorer,
                 crossoverProb=1, mutationProb=0.001, numSelected=2,
                 HUX=False, hybridModel="1010", hybridConstant=0.1):
        self.population = [Chromosome(chromosomeSize, scorer, HUX)
                           for _ in range(populationSize)]

        self.size = populationSize
        self.chromosomeSize = chromosomeSize
        self.mutationProb = mutationProb
        self.crossoverProb = crossoverProb
        self.numSelected = numSelected
        self.HUX = HUX

        if hybridModel == "1010":
            self.localImprovement = self.globalLocalImprovement
        elif hybridModel == "1001":
            self.localImprovement = self.randomLocalImprovement
        elif hybridModel == "1001M":
            self.localImprovement = self.elitistLocalImprovement
        else:
            raise ValueError('The hybrid model is unknown.'
                             'At the moment we only support "1010", "1001" and'
                             '"1001M".')

        self.generation = 1

    def binaryTournament(self):
        indices = np.random.choice(self.size, 2)
        contestants = [self.population[i] for i in indices]

        isFirstBest = contestants[0].score > contestants[1].score
        winner = contestants[0] if isFirstBest else contestants[1]
        return winner

    def selection(self):
        self.selected = [self.binaryTournament()
                         for _ in range(self.numSelected)]

    def recombination(self):
        # For every pair in selected chromosomes, do the crossover or maintain
        # the parents, based on the probability crossoverProb. It works both
        # on the stationary and generational paradigms. The chain.from_iterable
        # is necessary, as we are building a comprehensive list from pairs of
        # objects
        self.descendants = list(chain.from_iterable(
                            mother.crossover(father)
                            if np.random.uniform(0., 1.) < self.crossoverProb
                            else (mother, father)
                            for mother, father in pairwise(self.selected)))

    def mutation(self):
        numMutations = int(self.mutationProb * self.size * self.chromosomeSize)

        for _ in range(numMutations):
            chromosome = np.random.randint(self.size)
            gene = np.random.randint(self.chromosomeSize)

            self.population[chromosome].mutateGene(gene)

    def bestChromosome(self):
        # Sort in ascending order: first chromosome is the worst one
        self.population.sort(key=attrgetter('score'))

        return self.population[-1]

    def globalLocalImprovement(self):
        if self.generation % 10 == 0:
            for chromosome in self.population:
                chromosome.localImprovement()

    def randomLocalImprovement(self):
        numChanges = int(self.hybridConstant * self.size)
        chromosomesToChange = np.random.randint(0, self.size, numChanges)

        for chromosomeIdx in chromosomesToChange:
            self.population[chromosomeIdx].localImprovement()
        pass

    def elitistLocalImprovement(self):
        numChanges = int(self.hybridConstant * self.size)

        # Sort in descending order: first chromosome is the best one
        self.population.sort(key=attrgetter('score'), reverse=True)

        for chromosomeIdx in range(numChanges):
            self.population[chromosomeIdx].localImprovement()


class stationaryPopulation(Population):
    def __init__(self, chromosomeSize, scorer, HUX=False, populationSize=30,
                 crossoverProb=1, mutationProb=0.001, hybridModel="1010",
                 hybridConstant=0.1):
        super().__init__(populationSize, chromosomeSize, scorer,
                         crossoverProb, mutationProb, 2, HUX)

    def replacement(self):
        # Sort in ascending order: first chromosome is the worst one
        self.population.sort(key=attrgetter('score'))
        self.descendants.sort(key=attrgetter('score'), reverse=True)

        # Get the best chromosomes
        if self.descendants[0].score > self.population[0].score:
            self.population[0] = self.descendants[0]

        if self.descendants[1].score > self.population[1].score:
            self.population[1] = self.descendants[1]

        self.generation += 1


class generationalPopulation(Population):
    def __init__(self, chromosomeSize, scorer, HUX=False, populationSize=30,
                 crossoverProb=0.7, mutationProb=0.001, hybridModel="1010",
                 hybridConstant=0.1):
        super().__init__(populationSize, chromosomeSize, scorer, crossoverProb,
                         mutationProb, populationSize, HUX)

    def replacement(self):
        # Sort in descending order: first chromosome is the best one
        self.population.sort(key=attrgetter('score'), reverse=True)

        bestChromosome = self.population[0]

        self.population = self.descendants

        # Elitism
        if bestChromosome not in self.population:
            # Sort in ascending order: first chromosome is the worst one
            self.population.sort(key=attrgetter('score'))

            self.population[0] = bestChromosome

        self.generation += 1


def GA(train, target, scorer, stationary=True, HUX=False):
    def genScorer(chromosome):
        return scorer.scoreSolution(train[:, chromosome], target)

    # Number of features
    size = train.shape[1]

    if stationary:
        population = stationaryPopulation(size, genScorer, HUX)
    else:
        population = generationalPopulation(size, genScorer, HUX)

    # Evolution
    while(scorer.scoreCalls < 15000):
        population.selection()
        population.recombination()
        population.mutation()
        population.replacement()

    bestChromosome = population.bestChromosome()

    return bestChromosome.genes, bestChromosome.score


def stationaryGA(train, target, scorer):
    return GA(train, target, scorer, stationary=True)


def generationalGA(train, target, scorer):
    return GA(train, target, scorer, stationary=False)


def HUXstationaryGA(train, target, scorer):
    return GA(train, target, scorer, stationary=True, HUX=True)


def HUXgenerationalGA(train, target, scorer):
    return GA(train, target, scorer, stationary=False, HUX=True)
