import numpy as np
from ai.network import SnakeNet

class Genome:
    def __init__(self, weights: np.ndarray = None, fitness: float = 0.0):
        self.weights = weights.astype(np.float32) if weights is not None else None
        self.fitness = float(fitness)

    @classmethod
    def random(cls, num_weights: int, low: float = -1.0, high: float = 1.0):
        random_weights = np.random.uniform(low, high, size=num_weights).astype(np.float32)
        return cls(weights=random_weights)

    @classmethod
    def from_network(cls, net: SnakeNet):
        return cls(weights=net.get_weights_vector())

    def to_network(self, layer_sizes=None) -> SnakeNet:
        net = SnakeNet(layer_sizes)
        if self.weights is not None:
            net.set_weights_vector(self.weights)
        return net

    def clone(self):
        return Genome(weights=np.copy(self.weights), fitness=self.fitness)

    def save(self, filepath: str):
        np.save(filepath, self.weights)

    @classmethod
    def load(cls, filepath: str):
        if filepath.endswith(".npz"):
            data = np.load(filepath, allow_pickle=True)
            if "best_overall_weights" in data and data["best_overall_weights"].size > 0:
                weights = data["best_overall_weights"]
            elif "population_weights" in data:
                weights = data["population_weights"][0]
            else:
                raise ValueError(f"Nenhum vetor de pesos reconhecível encontrado em '{filepath}'.")
            fitness = float(data.get("best_overall_fitness", 0.0))
            return cls(weights=weights, fitness=fitness)

        weights = np.load(filepath)
        return cls(weights=weights)

    def __len__(self):
        return len(self.weights) if self.weights is not None else 0
