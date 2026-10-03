import random
import numpy as np
from ai.genome import Genome

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 100,
        num_weights: int = 951,
        elite_ratio: float = 0.03,
        tournament_size: int = 5,
        mutation_rate: float = 0.05,
        mutation_sigma: float = 0.20,
        min_mutation_rate: float = 0.01,
        min_mutation_sigma: float = 0.05
    ):
        self.population_size = population_size
        self.num_weights = num_weights
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size

        self.initial_mutation_rate = mutation_rate
        self.initial_mutation_sigma = mutation_sigma
        self.min_mutation_rate = min_mutation_rate
        self.min_mutation_sigma = min_mutation_sigma

    def init_population(self):
        return [Genome.random(self.num_weights) for _ in range(self.population_size)]

    def tournament_selection(self, population: list[Genome]) -> Genome:
        competitors = random.sample(population, self.tournament_size)
        return max(competitors, key=lambda ind: ind.fitness)

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        mask = np.random.rand(len(parent1)) < 0.5
        child = np.where(mask, parent1, parent2).astype(np.float32)
        return child

    def mutate(self, genome_weights: np.ndarray, mutation_rate: float, sigma: float) -> np.ndarray:
        child = np.copy(genome_weights)
        mask = np.random.rand(len(child)) < mutation_rate
        noise = np.random.normal(0.0, sigma, size=len(child)).astype(np.float32)
        child[mask] += noise[mask]
        return child

    def get_mutation_params(self, current_gen: int, max_gens: int):
        if max_gens <= 1:
            return self.initial_mutation_rate, self.initial_mutation_sigma

        progress = min(1.0, current_gen / max_gens)
        curr_rate = self.initial_mutation_rate - progress * (self.initial_mutation_rate - self.min_mutation_rate)
        curr_sigma = self.initial_mutation_sigma - progress * (self.initial_mutation_sigma - self.min_mutation_sigma)
        return curr_rate, curr_sigma

    def evolve_population(self, population: list[Genome], current_gen: int = 0, max_gens: int = 100):
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        fitnesses = [ind.fitness for ind in population]
        stats = {
            "best_fitness": fitnesses[0],
            "mean_fitness": float(np.mean(fitnesses)),
            "worst_fitness": fitnesses[-1],
            "best_genome": population[0].clone()
        }

        num_elites = max(1, int(self.population_size * self.elite_ratio))
        next_gen = [ind.clone() for ind in population[:num_elites]]

        curr_rate, curr_sigma = self.get_mutation_params(current_gen, max_gens)

        while len(next_gen) < self.population_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            child_weights = self.uniform_crossover(parent1.weights, parent2.weights)
            child_weights = self.mutate(child_weights, curr_rate, curr_sigma)

            next_gen.append(Genome(weights=child_weights))

        return next_gen, stats
