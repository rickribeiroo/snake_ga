from game.snake_env import SnakeEnv
from ai.perception import SnakePerception
from training.fitness import compute_fitness
import numpy as np

try:
    from numba import njit
    from ai.perception import _numba_features_single, NUMBA_PERCEPTION_AVAILABLE

    @njit(fastmath=True)
    def _numba_forward(x, w1, b1, w2, b2, w3, b3):
        h1 = np.empty(20, dtype=np.float32)
        for j in range(20):
            s = b1[j]
            for i in range(32):
                s += w1[j, i] * x[i]
            h1[j] = np.tanh(s)

        h2 = np.empty(12, dtype=np.float32)
        for j in range(12):
            s = b2[j]
            for i in range(20):
                s += w2[j, i] * h1[i]
            h2[j] = np.tanh(s)

        best_act = 0
        max_val = -1e9
        for j in range(3):
            s = b3[j]
            for i in range(12):
                s += w3[j, i] * h2[i]
            if s > max_val:
                max_val = s
                best_act = j

        return best_act

    @njit(fastmath=True)
    def _numba_predict_action(grid, hx, hy, dx, dy, fx, fy, snake_len, width, height, w1, b1, w2, b2, w3, b3):
        feats = np.empty(32, dtype=np.float32)
        _numba_features_single(grid, hx, hy, dx, dy, fx, fy, snake_len, width, height, feats)
        return _numba_forward(feats, w1, b1, w2, b2, w3, b3)

    _dummy_x = np.zeros(32, dtype=np.float32)
    _dummy_w1 = np.zeros((20, 32), dtype=np.float32)
    _dummy_b1 = np.zeros(20, dtype=np.float32)
    _dummy_w2 = np.zeros((12, 20), dtype=np.float32)
    _dummy_b2 = np.zeros(12, dtype=np.float32)
    _dummy_w3 = np.zeros((3, 12), dtype=np.float32)
    _dummy_b3 = np.zeros(3, dtype=np.float32)
    _ = _numba_forward(_dummy_x, _dummy_w1, _dummy_b1, _dummy_w2, _dummy_b2, _dummy_w3, _dummy_b3)

    if NUMBA_PERCEPTION_AVAILABLE:
        _dummy_grid = np.zeros((15, 15), dtype=np.int8)
        _ = _numba_predict_action(_dummy_grid, 0, 0, 1, 0, 5, 5, 3, 15, 15, _dummy_w1, _dummy_b1, _dummy_w2, _dummy_b2, _dummy_w3, _dummy_b3)
        NUMBA_FUSED_AVAILABLE = True
    else:
        NUMBA_FUSED_AVAILABLE = False

    NUMBA_FORWARD_AVAILABLE = True
except Exception:
    NUMBA_FORWARD_AVAILABLE = False
    NUMBA_FUSED_AVAILABLE = False


def evaluate_genome_single_episode(genome, width: int = 15, height: int = 15, unpacked_weights=None):
    env = SnakeEnv(width, height)
    done = False
    info = {}

    if NUMBA_FORWARD_AVAILABLE and genome.weights is not None:
        if unpacked_weights is None:
            w = genome.weights
            w1 = w[0:640].reshape(20, 32)
            b1 = w[640:660]
            w2 = w[660:900].reshape(12, 20)
            b2 = w[900:912]
            w3 = w[912:948].reshape(3, 12)
            b3 = w[948:951]
        else:
            w1, b1, w2, b2, w3, b3 = unpacked_weights

        if NUMBA_FUSED_AVAILABLE and hasattr(env, "grid"):
            grid = env.grid
            w_env = env.width
            h_env = env.height
            while not done:
                hx, hy = env.snake[0]
                dx, dy = env.direction
                fx, fy = env.food if env.food else (-1, -1)
                action = _numba_predict_action(grid, hx, hy, dx, dy, fx, fy, len(env.snake), w_env, h_env, w1, b1, w2, b2, w3, b3)
                _, _, done, info = env.step(action)
        else:
            while not done:
                obs = SnakePerception.get_features(env)
                action = _numba_forward(obs, w1, b1, w2, b2, w3, b3)
                _, _, done, info = env.step(action)
    else:
        brain = genome.to_network()
        while not done:
            obs = SnakePerception.get_features(env)
            action = brain.predict_action(obs)
            _, _, done, info = env.step(action)

    reason = info.get("reason", "unknown")
    fitness = compute_fitness(
        steps=env.steps,
        score=env.score,
        grid_width=width,
        grid_height=height,
        reason=reason,
        won=env.won,
        distance_reward=env.distance_reward
    )

    return fitness, env.score, env.steps, reason


def evaluate_genome_multi_grid(genome, grid_sizes=((10, 10), (15, 15)), trials: int = 1):
    total_fitness = 0.0
    total_score = 0
    total_steps = 0
    runs = len(grid_sizes) * max(1, trials)

    unpacked_weights = None
    if NUMBA_FORWARD_AVAILABLE and genome.weights is not None:
        w = genome.weights
        unpacked_weights = (
            w[0:640].reshape(20, 32),
            w[640:660],
            w[660:900].reshape(12, 20),
            w[900:912],
            w[912:948].reshape(3, 12),
            w[948:951]
        )

    for _ in range(max(1, trials)):
        for w, h in grid_sizes:
            fit, score, steps, _ = evaluate_genome_single_episode(genome, w, h, unpacked_weights=unpacked_weights)
            total_fitness += fit
            total_score += score
            total_steps += steps

    avg_fitness = total_fitness / runs
    return avg_fitness, total_score, total_steps
