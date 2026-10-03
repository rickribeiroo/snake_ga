import os
import sys
import time
import argparse
import pygame

from game.snake_env import SnakeEnv
from game.renderer import SnakeRenderer
from ai.perception import SnakePerception
from ai.genome import Genome

def play_best_ai(
    checkpoint_path: str = "checkpoints/best_overall.npy",
    grid_width: int = 15,
    grid_height: int = 15,
    cell_size: int = 40,
    fps: int = 12,
    show_rays: bool = True,
    shield: bool = False
):
    if not os.path.exists(checkpoint_path):
        print(f"[ERRO] Checkpoint não encontrado em: {checkpoint_path}")
        print("Treine um modelo primeiro executando: python -m training.train")
        sys.exit(1)

    print(f"Carregando genoma de: {checkpoint_path}...")
    genome = Genome.load(checkpoint_path)
    brain = genome.to_network()

    env = SnakeEnv(grid_width, grid_height, shield=shield)
    renderer = SnakeRenderer(grid_width, grid_height, cell_size=cell_size)

    print("=" * 60)
    print(f"ASSISTINDO IA JOGANDO SNAKE")
    print(f"Grid: {grid_width}x{grid_height} | FPS Inicial: {fps}")
    print("Controles:")
    print("  [ESPAÇO] : Pausar / Continuar")
    print("  [CIMA]   : Aumentar velocidade (FPS)")
    print("  [BAIXO]  : Diminuir velocidade (FPS)")
    print("  [V]      : Ligar / Desligar Raios de Visão")
    print("  [R]      : Reiniciar partida imediatamente")
    print("  [ESC]    : Sair")
    print("=" * 60)

    running = True
    paused = False
    current_fps = fps

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_rays = not show_rays
                elif event.key == pygame.K_UP:
                    current_fps = min(60, current_fps + 4)
                    print(f"FPS ajustado para: {current_fps}")
                elif event.key == pygame.K_DOWN:
                    current_fps = max(2, current_fps - 4)
                    print(f"FPS ajustado para: {current_fps}")
                elif event.key == pygame.K_r:
                    env.reset()

        if not paused:
            if not env.game_over:
                features = SnakePerception.get_features(env)
                action = brain.predict_action(features)
                env.step(action)
            else:
                time.sleep(1.2)
                env.reset()

        rays = SnakePerception.get_ray_endpoints(env, cell_size, renderer.HEADER_HEIGHT) if show_rays else None
        renderer.render(env, vision_rays=rays)
        renderer.clock.tick(current_fps)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assista a IA jogar Snake.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_overall.npy", help="Caminho do arquivo .npy")
    parser.add_argument("--width", type=int, default=15, help="Largura da grade")
    parser.add_argument("--height", type=int, default=15, help="Altura da grade")
    parser.add_argument("--fps", type=int, default=12, help="Velocidade da simulação em FPS")
    parser.add_argument("--cell-size", type=int, default=40, help="Tamanho de cada célula em pixels")
    parser.add_argument("--no-rays", action="store_true", help="Desativa o desenho dos raios")
    parser.add_argument("--shield", action="store_true", help="Ativa o filtro de sobrevivência (action masking)")

    args = parser.parse_args()

    play_best_ai(
        checkpoint_path=args.checkpoint,
        grid_width=args.width,
        grid_height=args.height,
        cell_size=args.cell_size,
        fps=args.fps,
        show_rays=not args.no_rays,
        shield=args.shield
    )
