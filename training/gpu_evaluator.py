import torch
import numpy as np
from game.snake_env import SnakeEnv
from ai.perception import SnakePerception
from training.fitness import compute_fitness
from ai.genome import Genome

class GPUPopulationEvaluator:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _unpack_population_weights(self, population: list[Genome]):
        N = len(population)
        all_weights = np.stack([ind.weights for ind in population])
        tensor_weights = torch.tensor(all_weights, dtype=torch.float32, device=self.device)

        w1_raw = tensor_weights[:, 0:640].view(N, 20, 32)
        W1 = w1_raw.transpose(1, 2)
        b1 = tensor_weights[:, 640:660].view(N, 1, 20)

        w2_raw = tensor_weights[:, 660:900].view(N, 12, 20)
        W2 = w2_raw.transpose(1, 2)
        b2 = tensor_weights[:, 900:912].view(N, 1, 12)

        w3_raw = tensor_weights[:, 912:948].view(N, 3, 12)
        W3 = w3_raw.transpose(1, 2)
        b3 = tensor_weights[:, 948:951].view(N, 1, 3)

        return W1, b1, W2, b2, W3, b3

    def evaluate_population(self, population: list[Genome], grid_sizes=((10, 10), (15, 15)), verbose: bool = True, trials: int = 1):
        N = len(population)
        W1, b1, W2, b2, W3, b3 = self._unpack_population_weights(population)

        total_fitness = np.zeros(N, dtype=np.float32)
        total_scores = np.zeros(N, dtype=np.int32)
        total_steps = np.zeros(N, dtype=np.int32)

        total_runs = len(grid_sizes) * trials

        for width, height in grid_sizes:
            for trial in range(trials):
                trial_str = f" [Trial {trial + 1}/{trials}]" if trials > 1 else ""
                envs = [SnakeEnv(width, height, shield=False) for _ in range(N)]
                active_indices = list(range(N))
                step_count = 0

                while active_indices:
                    step_count += 1

                    obs_np = SnakePerception.get_batch_features(envs, active_indices, width, height)
                    X = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(1)

                    if len(active_indices) == N:
                        sub_W1, sub_b1 = W1, b1
                        sub_W2, sub_b2 = W2, b2
                        sub_W3, sub_b3 = W3, b3
                    else:
                        idx_tensor = torch.as_tensor(active_indices, dtype=torch.long, device=self.device)
                        sub_W1 = W1[idx_tensor]
                        sub_b1 = b1[idx_tensor]
                        sub_W2 = W2[idx_tensor]
                        sub_b2 = b2[idx_tensor]
                        sub_W3 = W3[idx_tensor]
                        sub_b3 = b3[idx_tensor]

                    with torch.no_grad():
                        H1 = torch.tanh(torch.bmm(X, sub_W1) + sub_b1)
                        H2 = torch.tanh(torch.bmm(H1, sub_W2) + sub_b2)
                        Out = torch.bmm(H2, sub_W3) + sub_b3
                        actions = torch.argmax(Out.squeeze(1), dim=1).cpu().numpy()

                    still_active = []
                    for real_idx, action in zip(active_indices, actions):
                        env = envs[real_idx]
                        env.step(int(action))
                        if not env.game_over:
                            still_active.append(real_idx)

                    active_indices = still_active

                    if verbose and (step_count % 5 == 0 or len(active_indices) == 0):
                        max_current_score = max(env.score for env in envs)
                        print(
                            f"  [GPU] Grid {width}x{height}{trial_str} | Passo {step_count:3d} | "
                            f"Vivas: {len(active_indices):3d}/{N} | Max Macas: {max_current_score:2d}",
                            end="\r",
                            flush=True
                        )

                if verbose:
                    max_final_score = max(env.score for env in envs)
                    print(
                        f"  [GPU] Grid {width}x{height}{trial_str} Finalizado | Passos Totais: {step_count:3d} | "
                        f"Max Macas: {max_final_score:2d}                    ",
                        flush=True
                    )

                for i, env in enumerate(envs):
                    fit = compute_fitness(
                        steps=env.steps,
                        score=env.score,
                        grid_width=width,
                        grid_height=height,
                        reason=getattr(env, "death_reason", "collision"),
                        won=env.won,
                        distance_reward=env.distance_reward
                    )
                    total_fitness[i] += fit
                    total_scores[i] += env.score
                    total_steps[i] += env.steps

        results = []
        for i in range(N):
            avg_fit = float(total_fitness[i] / total_runs)
            population[i].fitness = avg_fit
            results.append((avg_fit, int(total_scores[i]), int(total_steps[i])))

        return results
