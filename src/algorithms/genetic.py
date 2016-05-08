from algorithms.utils import flip, genInitSolution, pairwise

from operator import attrgetter
import numpy as np
from copy import deepcopy
from itertools import chain

class Chromosome:
    def __init__(self, chromosomSize, scorer):
        self.size = chromosomSize
        self.genes = genInitSolution(self.size)

        self.scorer = scorer
        self.setScore()

    def setScore(self):
        self.score = self.scorer(self.genes)

    def mutateGene(self, gene=None):
        if gene is None:
            gene = np.random.randint(self.size)

        # Mutation!
        flip(self.genes, gene)

        # Update scoring
        self.setScore()

    def crossover(self, partner):
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


# Stationary
class Population:
    def __init__(self, populationSize, chromosomeSize, scorer,
                 crossoverProb = 1, mutationProb = 0.001):
        self.population = [Chromosome(chromosomeSize, scorer)
                           for _ in range(populationSize)]

        self.size = populationSize
        self.chromosomeSize = chromosomeSize
        self.mutationProb = mutationProb
        self.crossoverProb = crossoverProb

        self.generation = 1

    def binaryTournament(self):
        indices = np.random.choice(self.size, 2)
        contestants = [self.population[i] for i in indices]

        isFirstBest = contestants[0].score > contestants[1].score
        winner = contestants[0] if isFirstBest else contestants[1]
        return winner

    def selection(self):
        self.selected = [self.binaryTournament() for _ in range(2)]

    def generationalSelection(self):
        self.selected = [self.binaryTournament() for _ in range(self.size)]

    def recombination(self):
        mother = self.selected[0]
        father = self.selected[1]

        self.descendants = mother.crossover(father)

    # TODO: This is the saaaaaaaaaaaaame as above (replace it later)
    def generationalRecombination(self):
        self.descendants = list(chain.from_iterable(
                            mother.crossover(father)
                            if np.random.uniform(0., 1.) > self.crossoverProb
                            else (mother, father)
                            for mother, father in pairwise(self.selected)))

    def mutation(self):
        numMutations = int(self.mutationProb * self.size * self.chromosomeSize)

        for _ in range(numMutations):
            chromosome = np.random.randint(self.size)
            gene = np.random.randint(self.chromosomeSize)

            self.population[chromosome].mutateGene(gene)

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

    def generationalReplacement(self):
        # Sort in descending order: first chromosome is the best one
        self.population.sort(key=attrgetter('score'), reverse=True)

        bestChromosome = self.population[0]

        self.population = self.descendants

        # Elitism
        if bestChromosome not in self.population:
            # Sort in ascending order: first chromosome is the worst one
            self.population.sort(key=attrgetter('score'))

            self.population[0] = bestChromosome

    def bestChromosome(self):
        # Sort in ascending order: first chromosome is the worst one
        self.population.sort(key=attrgetter('score'))

        return self.population[-1]


def stationaryGA(train, target, scorer):
    def genScorer(chromosome):
        return scorer.scoreSolution(train[:, chromosome], target)

    # Number of features
    size = train.shape[1]

    # Initialization of the population
    population = Population(30, size, genScorer)

    # Evolition
    while(scorer.scoreCalls < 15000):
        population.selection()
        population.recombination()
        population.mutation()
        population.replacement()

    bestChromosome = population.bestChromosome()

    return bestChromosome.genes, bestChromosome.score

def stationaryGA(train, target, scorer):
    def genScorer(chromosome):
        return scorer.scoreSolution(train[:, chromosome], target)

    # Number of features
    size = train.shape[1]

    # Initialization of the population
    population = Population(30, size, genScorer)

    # Evolition
    while(scorer.scoreCalls < 15000):
        population.selection()
        population.recombination()
        population.mutation()
        population.replacement()

    bestChromosome = population.bestChromosome()

    return bestChromosome.genes, bestChromosome.score


def generationalGA(train, target, scorer):
    def genScorer(chromosome):
        return scorer.scoreSolution(train[:, chromosome], target)

    # Number of features
    size = train.shape[1]

    # Initialization of the population
    population = Population(30, size, genScorer, crossoverProb=0.7)

    # Evolition
    while(scorer.scoreCalls < 15000):
        population.generationalSelection()
        population.generationalRecombination()
        population.mutation()
        population.generationalReplacement()

    bestChromosome = population.bestChromosome()

    return bestChromosome.genes, bestChromosome.score
