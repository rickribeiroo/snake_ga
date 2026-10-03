import pygame
import sys
from game.snake_env import SnakeEnv
from ai.perception import SnakePerception

class SnakeRenderer:
    COLOR_BG = (22, 27, 34)
    COLOR_GRID = (33, 38, 45)
    COLOR_HEADER = (13, 17, 23)
    COLOR_HEAD = (46, 204, 113)
    COLOR_EYE = (15, 23, 42)
    COLOR_BODY = (39, 174, 96)
    COLOR_FOOD = (231, 76, 60)
    COLOR_TEXT = (240, 246, 252)
    COLOR_TEXT_DIM = (139, 148, 158)
    COLOR_RAY = (56, 189, 248, 80)

    HEADER_HEIGHT = 60

    def __init__(self, width=15, height=15, cell_size=64):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + self.HEADER_HEIGHT

        pygame.init()
        pygame.display.set_caption("Snake AI - Algoritmo Genético")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 13)
        self.font_game_over = pygame.font.SysFont("Segoe UI", 32, bold=True)

    def render(self, env, vision_rays=None):
        self.screen.fill(self.COLOR_BG)

        self._draw_header(env)
        self._draw_grid()

        if vision_rays:
            self._draw_rays(env, vision_rays)

        if env.food:
            self._draw_food(env.food)

        self._draw_snake(env)

        if env.game_over:
            self._draw_game_over(env)

        pygame.display.flip()

    def _draw_header(self, env):
        header_rect = pygame.Rect(0, 0, self.screen_width, self.HEADER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_HEADER, header_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.HEADER_HEIGHT), (self.screen_width, self.HEADER_HEIGHT), 2)

        score_text = self.font_main.render(f"Maçãs: {env.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (16, 8))

        steps_text = self.font_small.render(f"Passos: {env.steps}", True, self.COLOR_TEXT_DIM)
        stamina_left = max(0, env.max_steps_without_food - env.steps_since_last_food)
        stamina_text = self.font_small.render(f"Fôlego: {stamina_left}", True, self.COLOR_TEXT_DIM)

        self.screen.blit(steps_text, (16, 34))
        self.screen.blit(stamina_text, (130, 34))

        grid_info = self.font_small.render(f"Grid: {env.width}x{env.height}", True, self.COLOR_TEXT_DIM)
        self.screen.blit(grid_info, (self.screen_width - 85, 10))

    def _draw_grid(self):
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.HEADER_HEIGHT), (x, self.screen_height), 1)
        for y in range(self.HEADER_HEIGHT, self.screen_height, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y), 1)

    def _draw_food(self, food_pos):
        fx, fy = food_pos
        center_x = fx * self.cell_size + self.cell_size // 2
        center_y = fy * self.cell_size + self.HEADER_HEIGHT + self.cell_size // 2
        radius = self.cell_size // 2 - 3
        pygame.draw.circle(self.screen, self.COLOR_FOOD, (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (255, 150, 140), (center_x - 3, center_y - 3), radius // 3)

    def _draw_snake(self, env):
        for i, (x, y) in enumerate(env.snake):
            if i == 0:
                continue
            px = x * self.cell_size + 2
            py = y * self.cell_size + self.HEADER_HEIGHT + 2
            rect = pygame.Rect(px, py, self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.screen, self.COLOR_BODY, rect, border_radius=5)

        hx, hy = env.snake[0]
        h_px = hx * self.cell_size + 1
        h_py = hy * self.cell_size + self.HEADER_HEIGHT + 1
        head_rect = pygame.Rect(h_px, h_py, self.cell_size - 2, self.cell_size - 2)
        pygame.draw.rect(self.screen, self.COLOR_HEAD, head_rect, border_radius=7)

        self._draw_eyes(hx, hy, env.direction)

    def _draw_eyes(self, hx, hy, direction):
        dx, dy = direction
        cx = hx * self.cell_size + self.cell_size // 2
        cy = hy * self.cell_size + self.HEADER_HEIGHT + self.cell_size // 2
        eye_radius = max(2, self.cell_size // 10)
        offset = self.cell_size // 4

        if dx == 1:
            e1 = (cx + offset, cy - offset)
            e2 = (cx + offset, cy + offset)
        elif dx == -1:
            e1 = (cx - offset, cy - offset)
            e2 = (cx - offset, cy + offset)
        elif dy == -1:
            e1 = (cx - offset, cy - offset)
            e2 = (cx + offset, cy - offset)
        else:
            e1 = (cx - offset, cy + offset)
            e2 = (cx + offset, cy + offset)

        pygame.draw.circle(self.screen, self.COLOR_EYE, e1, eye_radius)
        pygame.draw.circle(self.screen, self.COLOR_EYE, e2, eye_radius)

    def _draw_rays(self, env, vision_rays):
        hx, hy = env.snake[0]
        start_pos = (
            hx * self.cell_size + self.cell_size // 2,
            hy * self.cell_size + self.HEADER_HEIGHT + self.cell_size // 2
        )
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        for target_pos in vision_rays:
            pygame.draw.line(overlay, self.COLOR_RAY, start_pos, target_pos, 2)
        self.screen.blit(overlay, (0, 0))

    def _draw_game_over(self, env):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title = "VITÓRIA!" if env.won else "GAME OVER"
        color = (46, 204, 113) if env.won else (231, 76, 60)
        text_surf = self.font_game_over.render(title, True, color)
        rect = text_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 20))
        self.screen.blit(text_surf, rect)

        sub_text = self.font_small.render("Pressione R para reiniciar ou ESC para sair", True, self.COLOR_TEXT)
        sub_rect = sub_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 25))
        self.screen.blit(sub_text, sub_rect)

    def handle_manual_input(self, env):
        curr_idx = env.DIRECTIONS.index(env.direction)
        desired_action = env.ACTION_STRAIGHT

        keys = pygame.key.get_pressed()
        target_dir = None

        if keys[pygame.K_UP]:
            target_dir = (0, -1)
        elif keys[pygame.K_RIGHT]:
            target_dir = (1, 0)
        elif keys[pygame.K_DOWN]:
            target_dir = (0, 1)
        elif keys[pygame.K_LEFT]:
            target_dir = (-1, 0)

        if keys[pygame.K_a]:
            return env.ACTION_LEFT
        if keys[pygame.K_d]:
            return env.ACTION_RIGHT
        if keys[pygame.K_w]:
            return env.ACTION_STRAIGHT

        if target_dir is not None:
            target_idx = env.DIRECTIONS.index(target_dir)
            diff = (target_idx - curr_idx) % 4

            if diff == 0:
                desired_action = env.ACTION_STRAIGHT
            elif diff == 1:
                desired_action = env.ACTION_RIGHT
            elif diff == 3:
                desired_action = env.ACTION_LEFT
            elif diff == 2:
                desired_action = env.ACTION_STRAIGHT

        return desired_action

    def run_manual_game(self, fps=10, show_rays=True):
        env = SnakeEnv(self.width, self.height)
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and env.game_over:
                        env.reset()
                    elif event.key == pygame.K_v:
                        show_rays = not show_rays

            if not env.game_over:
                action = self.handle_manual_input(env)
                env.step(action)

            vision_rays = SnakePerception.get_ray_endpoints(env, self.cell_size, self.HEADER_HEIGHT) if show_rays else None
            self.render(env, vision_rays=vision_rays)
            self.clock.tick(fps)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    renderer = SnakeRenderer(width=15, height=15, cell_size=32)
    renderer.run_manual_game(fps=10, show_rays=True)
