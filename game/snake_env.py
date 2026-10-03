import random
from collections import deque
import numpy as np
from utils.flood_fill import compute_free_space

class SnakeEnv:
    DIRECTIONS = [
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0)
    ]

    ACTION_STRAIGHT = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2

    def __init__(self, width=15, height=15, shield: bool = False):
        self.width = width
        self.height = height
        self.shield = shield
        self.reset()

    def reset(self, width=None, height=None):
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

        head_x = self.width // 2
        head_y = self.height // 2
        self.direction = random.choice(self.DIRECTIONS)
        dx, dy = self.direction
        self.snake = deque([
            (head_x, head_y),
            (head_x - dx, head_y - dy),
            (head_x - 2 * dx, head_y - 2 * dy)
        ])

        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        for sx, sy in self.snake:
            self.grid[sy, sx] = 1
        self.score = 0
        self.steps = 0
        self.steps_since_last_food = 0
        self.max_steps_without_food = max(150, self.width * self.height) + len(self.snake)
        self.game_over = False
        self.won = False
        self.death_reason = ""
        self.distance_reward = 0.0
        self.consecutive_rights = 0
        self.consecutive_lefts = 0
        self.visited_states = set()
        self.food = self._place_food()

        return None

    def _place_food(self):
        if len(self.snake) >= self.width * self.height:
            return None

        empty_count = (self.width * self.height) - len(self.snake)
        if empty_count < 30:
            empty_cells = np.argwhere(self.grid == 0)
            idx = random.randint(0, len(empty_cells) - 1)
            return (int(empty_cells[idx][1]), int(empty_cells[idx][0]))

        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.grid[y, x] == 0:
                return (x, y)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]

        x, y = pt

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        if pt != self.snake[0]:
            return self.grid[y, x] == 1

        return any(s == pt for i, s in enumerate(self.snake) if i > 0)

    def step(self, action):
        if self.game_over:
            return None, 0.0, True, {"reason": "already_dead"}

        self.steps += 1

        prev_dist = 0
        if self.food:
            prev_dist = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])

        if action == self.ACTION_RIGHT:
            self.consecutive_rights += 1
            self.consecutive_lefts = 0
        elif action == self.ACTION_LEFT:
            self.consecutive_lefts += 1
            self.consecutive_rights = 0
        else:
            self.consecutive_rights = 0
            self.consecutive_lefts = 0

        if self.consecutive_rights >= 4 or self.consecutive_lefts >= 4:
            self.distance_reward -= 15.0
            self.consecutive_rights = 0
            self.consecutive_lefts = 0

        curr_idx = self.DIRECTIONS.index(self.direction)
        if action == self.ACTION_STRAIGHT:
            new_idx = curr_idx
        elif action == self.ACTION_LEFT:
            new_idx = (curr_idx - 1) % 4
        elif action == self.ACTION_RIGHT:
            new_idx = (curr_idx + 1) % 4
        else:
            raise ValueError(f"Ação inválida: {action}. Use 0 (reto), 1 (esquerda) ou 2 (direita).")

        cand_dir = self.DIRECTIONS[new_idx]
        head_x, head_y = self.snake[0]
        new_head = (head_x + cand_dir[0], head_y + cand_dir[1])

        if self.shield and self.is_collision(new_head):
            safe_alternatives = []
            for alt_action in (self.ACTION_STRAIGHT, self.ACTION_LEFT, self.ACTION_RIGHT):
                if alt_action == action:
                    continue
                if alt_action == self.ACTION_STRAIGHT:
                    alt_idx = curr_idx
                elif alt_action == self.ACTION_LEFT:
                    alt_idx = (curr_idx - 1) % 4
                else:
                    alt_idx = (curr_idx + 1) % 4
                alt_dir = self.DIRECTIONS[alt_idx]
                alt_head = (head_x + alt_dir[0], head_y + alt_dir[1])
                if not self.is_collision(alt_head):
                    safe_alternatives.append((alt_action, alt_idx, alt_dir, alt_head))

            if safe_alternatives:
                self.distance_reward -= 3.0
                if len(safe_alternatives) == 1:
                    action, new_idx, cand_dir, new_head = safe_alternatives[0]
                else:
                    best_alt = safe_alternatives[0]
                    best_space = -1.0
                    for alt in safe_alternatives:
                        space = compute_free_space(self.width, self.height, self.snake, alt[3], grid=self.grid)
                        if space > best_space:
                            best_space = space
                            best_alt = alt
                    action, new_idx, cand_dir, new_head = best_alt

        self.direction = cand_dir

        if self.is_collision(new_head):
            self.game_over = True
            is_self = self.grid[new_head[1], new_head[0]] == 1 if (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height) else False
            self.death_reason = "self_collision" if is_self else "wall_collision"
            return None, -10.0, True, {
                "score": self.score,
                "steps": self.steps,
                "reason": self.death_reason
            }

        self.snake.appendleft(new_head)
        self.grid[new_head[1], new_head[0]] = 1

        head_state = (new_head[0], new_head[1], self.direction)
        if head_state in self.visited_states:
            self.distance_reward -= 8.0
        else:
            self.visited_states.add(head_state)

        if self.food:
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if new_dist < prev_dist:
                self.distance_reward += 1.5
            elif new_dist > prev_dist:
                self.distance_reward -= 2.0

        reward = 0.0
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.distance_reward += 25.0
            self.consecutive_rights = 0
            self.consecutive_lefts = 0
            self.visited_states.clear()
            self.steps_since_last_food = 0
            self.max_steps_without_food = max(150, self.width * self.height) + len(self.snake)

            if len(self.snake) >= self.width * self.height:
                self.won = True
                self.game_over = True
                self.death_reason = "won"
                return None, 100.0, True, {
                    "score": self.score,
                    "steps": self.steps,
                    "reason": "won"
                }

            self.food = self._place_food()
        else:
            tail = self.snake.pop()
            self.grid[tail[1], tail[0]] = 0
            self.steps_since_last_food += 1

        if self.steps_since_last_food >= self.max_steps_without_food:
            self.game_over = True
            self.death_reason = "starvation"
            return None, -5.0, True, {
                "score": self.score,
                "steps": self.steps,
                "reason": "starvation"
            }

        return None, reward, False, {
            "score": self.score,
            "steps": self.steps
        }

    def _get_obs(self):
        return {
            "snake": list(self.snake),
            "head": self.snake[0],
            "direction": self.direction,
            "food": self.food,
            "grid_size": (self.width, self.height),
            "score": self.score,
            "steps": self.steps
        }

    def render_ascii(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        if self.food:
            fx, fy = self.food
            grid[fy][fx] = "F"

        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                continue
            grid[y][x] = "o"

        hx, hy = self.snake[0]
        grid[hy][hx] = "H"

        lines = ["#" + "-" * self.width + "#"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("#" + "-" * self.width + "#")
        return "\n".join(lines)