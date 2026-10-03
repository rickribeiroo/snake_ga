import os
import sys
import time
import argparse
import numpy as np
from game.snake_env import SnakeEnv
from ai.perception import SnakePerception
from ai.genome import Genome

def load_genome_brain(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

    if checkpoint_path.endswith(".npy"):
        genome = Genome.load(checkpoint_path)
    elif checkpoint_path.endswith(".npz"):
        data = np.load(checkpoint_path, allow_pickle=True)
        if "best_overall_weights" in data and data["best_overall_weights"].size > 0:
            genome = Genome(weights=data["best_overall_weights"])
        elif "population_weights" in data:
            genome = Genome(weights=data["population_weights"][0])
        else:
            raise KeyError(f"Nenhum peso válido encontrado no arquivo {checkpoint_path}")
    else:
        raise ValueError(f"Formato não suportado: {checkpoint_path}. Use .npy ou .npz")

    return genome.to_network()

def run_single_benchmark_game(brain, width: int = 15, height: int = 15, shield: bool = False):
    env = SnakeEnv(width, height, shield=shield)
    shield_interventions = 0

    while not env.game_over:
        obs = SnakePerception.get_features(env)
        action = brain.predict_action(obs)

        if shield:
            curr_idx = env.DIRECTIONS.index(env.direction)
            if action == 0:
                cand_idx = curr_idx
            elif action == 1:
                cand_idx = (curr_idx - 1) % 4
            else:
                cand_idx = (curr_idx + 1) % 4
            cand_dir = env.DIRECTIONS[cand_idx]
            new_head = (env.snake[0][0] + cand_dir[0], env.snake[0][1] + cand_dir[1])
            if env.is_collision(new_head):
                shield_interventions += 1

        env.step(action)

    return {
        "score": int(env.score),
        "steps": int(env.steps),
        "death_reason": getattr(env, "death_reason", "unknown"),
        "won": bool(env.won),
        "shield_interventions": int(shield_interventions)
    }

def evaluate_model(brain, num_games: int = 100, width: int = 15, height: int = 15, shield: bool = False, model_name: str = "Modelo"):
    print(f"\n[BENCHMARK] Avaliando '{model_name}' em {num_games} partidas (Shield: {'LIGADO' if shield else 'DESLIGADO'})...")
    
    t0 = time.perf_counter()
    results = []
    
    for i in range(num_games):
        res = run_single_benchmark_game(brain, width=width, height=height, shield=shield)
        results.append(res)
        
        if (i + 1) % max(1, num_games // 10) == 0 or (i + 1) == num_games:
            pct = int(((i + 1) / num_games) * 100)
            bar = "=" * (pct // 5) + "-" * (20 - (pct // 5))
            print(f"  Progresso: [{bar}] {pct:3d}% ({i + 1}/{num_games})", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print()

    scores = np.array([r["score"] for r in results], dtype=np.int32)
    steps = np.array([r["steps"] for r in results], dtype=np.int32)
    reasons = [r["death_reason"] for r in results]
    shields = np.array([r["shield_interventions"] for r in results], dtype=np.int32)

    total = len(results)
    death_counts = {
        "self_collision": reasons.count("self_collision"),
        "wall_collision": reasons.count("wall_collision"),
        "starvation": reasons.count("starvation"),
        "won": sum(1 for r in results if r["won"])
    }

    buckets = {
        "0 - 10": int(np.sum((scores >= 0) & (scores <= 10))),
        "11 - 25": int(np.sum((scores >= 11) & (scores <= 25))),
        "26 - 40": int(np.sum((scores >= 26) & (scores <= 40))),
        "41 - 55": int(np.sum((scores >= 41) & (scores <= 55))),
        "56 - 70": int(np.sum((scores >= 56) & (scores <= 70))),
        "71+": int(np.sum(scores >= 71))
    }

    stats = {
        "name": model_name,
        "num_games": total,
        "elapsed_seconds": round(elapsed, 2),
        "fps_simulated": round(float(np.sum(steps)) / elapsed, 1) if elapsed > 0 else 0,
        "score_mean": round(float(np.mean(scores)), 2),
        "score_std": round(float(np.std(scores)), 2),
        "score_median": float(np.median(scores)),
        "score_min": int(np.min(scores)),
        "score_max": int(np.max(scores)),
        "score_p25": float(np.percentile(scores, 25)),
        "score_p75": float(np.percentile(scores, 75)),
        "steps_mean": round(float(np.mean(steps)), 1),
        "steps_median": float(np.median(steps)),
        "steps_min": int(np.min(steps)),
        "steps_max": int(np.max(steps)),
        "steps_per_apple": round(float(np.sum(steps)) / max(1, int(np.sum(scores))), 1),
        "death_counts": death_counts,
        "death_pct": {k: round((v / total) * 100, 1) for k, v in death_counts.items()},
        "buckets": buckets,
        "buckets_pct": {k: round((v / total) * 100, 1) for k, v in buckets.items()},
        "shield_active": shield,
        "shield_interventions_mean": round(float(np.mean(shields)), 1),
        "shield_interventions_max": int(np.max(shields))
    }

    return stats

def print_single_report(stats: dict):
    print("=" * 68)
    print(f" RELATÓRIO DE BENCHMARK: {stats['name'].upper()}")
    print("=" * 68)
    print(f" Partidas: {stats['num_games']} | Tempo: {stats['elapsed_seconds']}s ({stats['fps_simulated']} passos/segundo)")
    print(f" Modo Shield: {'ATIVADO (Intervenções rastreadas)' if stats['shield_active'] else 'DESATIVADO (Rede Neural 100% Pura)'}")
    print("-" * 68)
    print(" 1. PONTUAÇÃO (MAÇÃS):")
    print(f"    Média:   {stats['score_mean']} ± {stats['score_std']}")
    print(f"    Mediana: {stats['score_median']} (Faixa Central 50%: {stats['score_p25']} a {stats['score_p75']})")
    print(f"    Mínimo:  {stats['score_min']} maçãs")
    print(f"    Máximo:  {stats['score_max']} maçãs (Recorde)")
    print("-" * 68)
    print(" 2. SOBREVIVÊNCIA E EFICIÊNCIA:")
    print(f"    Passos Médios por Partida: {stats['steps_mean']} (Mín: {stats['steps_min']} | Máx: {stats['steps_max']})")
    print(f"    Eficiência de Rota:        {stats['steps_per_apple']} passos para cada maçã comida")
    if stats['shield_active']:
        print(f"    Intervenções do Shield:    Média de {stats['shield_interventions_mean']} salvamentos/partida (Máx: {stats['shield_interventions_max']})")
    print("-" * 68)
    print(" 3. CAUSAS DE TÉRMINO DA PARTIDA:")
    d = stats['death_pct']
    c = stats['death_counts']
    print(f"    Batida no Próprio Corpo (self_collision): {c['self_collision']:3d} ({d['self_collision']:5.1f}%)")
    print(f"    Batida na Parede (wall_collision):        {c['wall_collision']:3d} ({d['wall_collision']:5.1f}%)")
    print(f"    Inanição / Sem Comida (starvation):       {c['starvation']:3d} ({d['starvation']:5.1f}%)")
    print(f"    Vitória (tabuleiro 15x15 preenchido):     {c['won']:3d} ({d['won']:5.1f}%)")
    print("-" * 68)
    print(" 4. DISTRIBUIÇÃO DAS PONTUAÇÕES (HISTOGRAMA):")
    for bucket, count in stats['buckets'].items():
        pct = stats['buckets_pct'][bucket]
        bar = "#" * int(pct // 3)
        print(f"    {bucket:7s} maçãs: {count:3d} partidas ({pct:5.1f}%) | {bar}")
    print("=" * 68)

def print_comparison_table(stats_a: dict, stats_b: dict):
    print("\n" + "=" * 76)
    print(f" COMPARAÇÃO DIRETA LADO A LADO ({stats_a['num_games']} partidas cada)")
    print("=" * 76)
    name_a = stats_a['name'][:24]
    name_b = stats_b['name'][:24]
    print(f" Métrica                           | {name_a:24s} | {name_b:24s}")
    print("-" * 76)
    print(f" Modo Shield                       | {'LIGADO' if stats_a['shield_active'] else 'DESLIGADO':24s} | {'LIGADO' if stats_b['shield_active'] else 'DESLIGADO':24s}")
    print(f" Média de Maçãs                    | {stats_a['score_mean']:24.2f} | {stats_b['score_mean']:24.2f}")
    print(f" Mediana de Maçãs                  | {stats_a['score_median']:24.1f} | {stats_b['score_median']:24.1f}")
    print(f" Recorde (Máximo)                  | {stats_a['score_max']:24d} | {stats_b['score_max']:24d}")
    print(f" Pior Caso (Mínimo)                | {stats_a['score_min']:24d} | {stats_b['score_min']:24d}")
    print(f" Desvio Padrão                     | {stats_a['score_std']:24.2f} | {stats_b['score_std']:24.2f}")
    print(f" Passos Médios por Partida         | {stats_a['steps_mean']:24.1f} | {stats_b['steps_mean']:24.1f}")
    print(f" Passos por Maçã (Eficiência)      | {stats_a['steps_per_apple']:24.1f} | {stats_b['steps_per_apple']:24.1f}")
    print("-" * 76)
    print(" CAUSAS DE MORTE (%)               |                          |")
    da = stats_a['death_pct']
    db = stats_b['death_pct']
    print(f"   Corpo da Cobra (self_collision) | {da['self_collision']:23.1f}% | {db['self_collision']:23.1f}%")
    print(f"   Parede (wall_collision)         | {da['wall_collision']:23.1f}% | {db['wall_collision']:23.1f}%")
    print(f"   Inanição (starvation)           | {da['starvation']:23.1f}% | {db['starvation']:23.1f}%")
    print("-" * 76)
    print(" TAXAS DE SUCESSO                  |                          |")
    p30_a = sum(v for k, v in stats_a['buckets'].items() if k in ["26 - 40", "41 - 55", "56 - 70", "71+"])
    p30_b = sum(v for k, v in stats_b['buckets'].items() if k in ["26 - 40", "41 - 55", "56 - 70", "71+"])
    pct30_a = (p30_a / stats_a['num_games']) * 100
    pct30_b = (p30_b / stats_b['num_games']) * 100
    print(f"   Partidas com >= 26 maçãs        | {pct30_a:23.1f}% | {pct30_b:23.1f}%")
    
    p50_a = sum(v for k, v in stats_a['buckets'].items() if k in ["56 - 70", "71+"])
    p50_b = sum(v for k, v in stats_b['buckets'].items() if k in ["56 - 70", "71+"])
    pct50_a = (p50_a / stats_a['num_games']) * 100
    pct50_b = (p50_b / stats_b['num_games']) * 100
    print(f"   Partidas com >= 56 maçãs        | {pct50_a:23.1f}% | {pct50_b:23.1f}%")
    print("=" * 76)

def main():
    parser = argparse.ArgumentParser(description="Benchmark rigoroso para modelos de IA do Snake.")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/best_overall.npy", help="Caminho do arquivo .npy ou .npz")
    parser.add_argument("--compare", type=str, default=None, help="Caminho de um segundo modelo para comparar lado a lado")
    parser.add_argument("--games", "-n", type=int, default=100, help="Quantidade de partidas a simular (padrão: 100)")
    parser.add_argument("--shield", action="store_true", help="Ativa o shield durante a avaliação (padrão: desligado)")
    parser.add_argument("--width", type=int, default=15, help="Largura do grid (padrão: 15)")
    parser.add_argument("--height", type=int, default=15, help="Altura do grid (padrão: 15)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Caminho opcional para salvar relatório em JSON")

    args = parser.parse_args()

    brain_a = load_genome_brain(args.checkpoint)
    name_a = os.path.basename(os.path.dirname(args.checkpoint) or ".") + "/" + os.path.basename(args.checkpoint)
    stats_a = evaluate_model(brain_a, num_games=args.games, width=args.width, height=args.height, shield=args.shield, model_name=name_a)
    print_single_report(stats_a)

    if args.compare:
        brain_b = load_genome_brain(args.compare)
        name_b = os.path.basename(os.path.dirname(args.compare) or ".") + "/" + os.path.basename(args.compare)
        stats_b = evaluate_model(brain_b, num_games=args.games, width=args.width, height=args.height, shield=args.shield, model_name=name_b)
        print_single_report(stats_b)
        print_comparison_table(stats_a, stats_b)

    if args.output:
        import json
        out_data = {"model_a": stats_a}
        if args.compare:
            out_data["model_b"] = stats_b
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4)
        print(f"\n[OK] Relatório exportado com sucesso para: {args.output}")

if __name__ == "__main__":
    main()
