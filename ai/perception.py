import math
import numpy as np
from utils.flood_fill import compute_free_space

try:
    from numba import njit, prange
    from utils.flood_fill import _numba_bfs

    RAY_DIRS = np.array([
        [0, -1], [1, -1], [1, 0], [1, 1],
        [0, 1], [-1, 1], [-1, 0], [-1, -1]
    ], dtype=np.int32)

    DIRECTIONS = np.array([
        [0, -1], [1, 0], [0, 1], [-1, 0]
    ], dtype=np.int32)

    @njit(fastmath=True)
    def _numba_features_single(grid, hx, hy, dx, dy, fx, fy, snake_len, width, height, out_feats):
        for i in range(8):
            rdx = RAY_DIRS[i, 0]
            rdy = RAY_DIRS[i, 1]

            dist_x = (width - 1 - hx) if rdx > 0 else (hx if rdx < 0 else -1)
            dist_y = (height - 1 - hy) if rdy > 0 else (hy if rdy < 0 else -1)

            if dist_x >= 0 and dist_y >= 0:
                steps_to_wall = min(dist_x, dist_y)
                max_in_dir = min(width, height)
            elif dist_x >= 0:
                steps_to_wall = dist_x
                max_in_dir = width
            else:
                steps_to_wall = dist_y
                max_in_dir = height

            wall_dist = min(1.0, steps_to_wall / max_in_dir)
            body_dist = 1.0
            food_dist = 1.0
            found_body = False
            found_food = False

            cx = hx
            cy = hy
            for s in range(1, steps_to_wall + 1):
                cx += rdx
                cy += rdy
                if not found_body and grid[cy, cx] == 1:
                    body_dist = s / max_in_dir
                    found_body = True
                if not found_food and cx == fx and cy == fy:
                    food_dist = s / max_in_dir
                    found_food = True
                if found_body and found_food:
                    break

            out_feats[i * 3] = wall_dist
            out_feats[i * 3 + 1] = body_dist
            out_feats[i * 3 + 2] = food_dist

        out_feats[24] = float(dx)
        out_feats[25] = float(dy)

        if fx >= 0 and fy >= 0:
            angle_food = np.arctan2(fy - hy, fx - hx)
            angle_dir = np.arctan2(dy, dx)
            delta_angle = angle_food - angle_dir
            out_feats[26] = np.cos(delta_angle)
            out_feats[27] = np.sin(delta_angle)
        else:
            out_feats[26] = 0.0
            out_feats[27] = 0.0

        out_feats[28] = snake_len / (width * height)

        dir_idx = 0
        for d in range(4):
            if DIRECTIONS[d, 0] == dx and DIRECTIONS[d, 1] == dy:
                dir_idx = d
                break

        dir_straight = DIRECTIONS[dir_idx]
        dir_left = DIRECTIONS[(dir_idx - 1) % 4]
        dir_right = DIRECTIONS[(dir_idx + 1) % 4]

        out_feats[29] = _numba_bfs(grid, hx + dir_straight[0], hy + dir_straight[1], width, height)
        out_feats[30] = _numba_bfs(grid, hx + dir_left[0], hy + dir_left[1], width, height)
        out_feats[31] = _numba_bfs(grid, hx + dir_right[0], hy + dir_right[1], width, height)

    @njit(parallel=True, fastmath=True)
    def _numba_features_batch(grids, heads, dirs, foods, lengths, width, height, out_batch):
        N = grids.shape[0]
        for i in prange(N):
            _numba_features_single(
                grids[i], heads[i, 0], heads[i, 1],
                dirs[i, 0], dirs[i, 1],
                foods[i, 0], foods[i, 1],
                lengths[i], width, height,
                out_batch[i]
            )

    _dummy_grid = np.zeros((1, 15, 15), dtype=np.int8)
    _dummy_heads = np.zeros((1, 2), dtype=np.int32)
    _dummy_dirs = np.array([[1, 0]], dtype=np.int32)
    _dummy_foods = np.zeros((1, 2), dtype=np.int32)
    _dummy_lens = np.array([3], dtype=np.int32)
    _dummy_out = np.zeros((1, 32), dtype=np.float32)
    _numba_features_batch(_dummy_grid, _dummy_heads, _dummy_dirs, _dummy_foods, _dummy_lens, 15, 15, _dummy_out)
    NUMBA_PERCEPTION_AVAILABLE = True
except Exception:
    NUMBA_PERCEPTION_AVAILABLE = False


class SnakePerception:
    RAY_DIRECTIONS = [
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1)
    ]

    @classmethod
    def get_features(cls, env, out_feats=None):
        grid = getattr(env, "grid", None)
        if NUMBA_PERCEPTION_AVAILABLE and grid is not None:
            if out_feats is None:
                out_feats = np.empty(32, dtype=np.float32)
            hx, hy = env.snake[0]
            dx, dy = env.direction
            fx, fy = env.food if env.food else (-1, -1)
            _numba_features_single(
                grid, hx, hy, dx, dy, fx, fy,
                len(env.snake), env.width, env.height, out_feats
            )
            return out_feats

        features = []
        hx, hy = env.snake[0]
        snake_body = {pos for i, pos in enumerate(env.snake) if i > 0}

        for rdx, rdy in cls.RAY_DIRECTIONS:
            dist_x = (env.width - 1 - hx) if rdx > 0 else (hx if rdx < 0 else None)
            dist_y = (env.height - 1 - hy) if rdy > 0 else (hy if rdy < 0 else None)

            if dist_x is not None and dist_y is not None:
                steps_to_wall = min(dist_x, dist_y)
                max_in_dir = min(env.width, env.height)
            elif dist_x is not None:
                steps_to_wall = dist_x
                max_in_dir = env.width
            else:
                steps_to_wall = dist_y
                max_in_dir = env.height

            wall_dist = min(1.0, steps_to_wall / max_in_dir)
            body_dist = 1.0
            food_dist = 1.0
            found_body = False
            found_food = False

            cx, cy = hx, hy
            for s in range(1, steps_to_wall + 1):
                cx += rdx
                cy += rdy
                if not found_body and (cx, cy) in snake_body:
                    body_dist = s / max_in_dir
                    found_body = True
                if not found_food and env.food and (cx, cy) == env.food:
                    food_dist = s / max_in_dir
                    found_food = True
                if found_body and found_food:
                    break

            features.append(wall_dist)
            features.append(body_dist)
            features.append(food_dist)

        dx, dy = env.direction
        features.append(float(dx))
        features.append(float(dy))

        if env.food:
            fx, fy = env.food
            angle_food = math.atan2(fy - hy, fx - hx)
            angle_dir = math.atan2(dy, dx)
            delta_angle = angle_food - angle_dir
            features.append(math.cos(delta_angle))
            features.append(math.sin(delta_angle))
        else:
            features.append(0.0)
            features.append(0.0)

        snake_ratio = len(env.snake) / (env.width * env.height)
        features.append(snake_ratio)

        curr_idx = env.DIRECTIONS.index(env.direction)
        dir_straight = env.direction
        dir_left = env.DIRECTIONS[(curr_idx - 1) % 4]
        dir_right = env.DIRECTIONS[(curr_idx + 1) % 4]

        pos_straight = (hx + dir_straight[0], hy + dir_straight[1])
        pos_left = (hx + dir_left[0], hy + dir_left[1])
        pos_right = (hx + dir_right[0], hy + dir_right[1])

        free_straight = compute_free_space(env.width, env.height, snake_body, pos_straight, grid=grid)
        free_left = compute_free_space(env.width, env.height, snake_body, pos_left, grid=grid)
        free_right = compute_free_space(env.width, env.height, snake_body, pos_right, grid=grid)

        features.append(free_straight)
        features.append(free_left)
        features.append(free_right)

        return np.array(features, dtype=np.float32)

    @classmethod
    def get_batch_features(cls, envs, active_indices, width, height):
        N = len(active_indices)
        if N == 0:
            return np.empty((0, 32), dtype=np.float32)

        if NUMBA_PERCEPTION_AVAILABLE:
            grids = np.empty((N, height, width), dtype=np.int8)
            heads = np.empty((N, 2), dtype=np.int32)
            dirs = np.empty((N, 2), dtype=np.int32)
            foods = np.empty((N, 2), dtype=np.int32)
            lengths = np.empty(N, dtype=np.int32)
            out_batch = np.empty((N, 32), dtype=np.float32)

            for i, real_idx in enumerate(active_indices):
                env = envs[real_idx]
                grids[i] = env.grid
                h = env.snake[0]
                heads[i, 0] = h[0]
                heads[i, 1] = h[1]
                d = env.direction
                dirs[i, 0] = d[0]
                dirs[i, 1] = d[1]
                f = env.food
                if f:
                    foods[i, 0] = f[0]
                    foods[i, 1] = f[1]
                else:
                    foods[i, 0] = -1
                    foods[i, 1] = -1
                lengths[i] = len(env.snake)

            _numba_features_batch(grids, heads, dirs, foods, lengths, width, height, out_batch)
            return out_batch

        obs_list = [cls.get_features(envs[i]) for i in active_indices]
        return np.array(obs_list, dtype=np.float32)

    @classmethod
    def get_ray_endpoints(cls, env, cell_size=32, header_height=60):
        hx, hy = env.snake[0]
        endpoints = []

        for rdx, rdy in cls.RAY_DIRECTIONS:
            cx, cy = hx, hy
            while 0 <= cx + rdx < env.width and 0 <= cy + rdy < env.height:
                cx += rdx
                cy += rdy

            px = cx * cell_size + cell_size // 2
            py = cy * cell_size + header_height + cell_size // 2
            endpoints.append((px, py))

        return endpoints
