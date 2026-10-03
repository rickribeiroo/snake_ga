import os
import sys
import json
import time
import argparse
import multiprocessing as mp
import numpy as np
import torch
import matplotlib.pyplot as plt

from ai.ga import GeneticAlgorithm
from ai.genome import Genome
from training.evaluator import evaluate_genome_multi_grid
from training.gpu_evaluator import GPUPopulationEvaluator

def _evaluate_worker(args):
    weights, grid_sizes, trials = args
    genome = Genome(weights=weights)
    avg_fitness, total_score, total_steps = evaluate_genome_multi_grid(genome, grid_sizes, trials=trials)
    return avg_fitness, total_score, total_steps


class Trainer:
    def __init__(
        self,
        population_size: int = 300,
        generations: int = 150,
        grid_sizes: tuple = ((10, 10), (15, 15)),
        device: str = None,
        num_workers: int = None,
        checkpoint_dir: str = "checkpoints",
        trials: int = 1,
        save_every: int = 10,
        resume: str = None,
        from_gen: int = None,
        mutation_rate: float = 0.035,
        mutation_sigma: float = 0.15,
        fixed_mutation: bool = True,
        infinite: bool = False
    ):
        self.population_size = population_size
        self.infinite = infinite or (generations is None or generations <= 0)
        self.generations = 10000000 if self.infinite else generations
        self.grid_sizes = grid_sizes
        self.checkpoint_dir = checkpoint_dir
        self.trials = trials
        self.save_every = max(1, save_every)
        self.resume_target = resume or (str(from_gen) if from_gen is not None else None)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device.lower()

        self.num_workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        min_rate = mutation_rate if fixed_mutation else max(0.015, mutation_rate * 0.4)
        min_sigma = mutation_sigma if fixed_mutation else max(0.05, mutation_sigma * 0.4)

        self.ga = GeneticAlgorithm(
            population_size=self.population_size,
            num_weights=951,
            elite_ratio=0.03,
            tournament_size=5,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            min_mutation_rate=min_rate,
            min_mutation_sigma=min_sigma
        )

        self.history = {
            "best_fitness": [],
            "mean_fitness": [],
            "best_score": [],
            "mean_score": []
        }

        self.best_overall_fitness = -1.0
        self.best_overall_genome = None

        if self.device == "cuda":
            self.gpu_evaluator = GPUPopulationEvaluator(device="cuda")
        else:
            self.gpu_evaluator = None

    def plot_history(self):
        if not self.history["best_fitness"]:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        gens = range(len(self.history["best_fitness"]))

        best_plot = [x / 100.0 for x in self.history["best_fitness"]]
        mean_plot = [x / 100.0 for x in self.history["mean_fitness"]]

        ax1.plot(gens, best_plot, label="Melhor Fitness (/ 10M)", color="#2ecc71", linewidth=2)
        ax1.plot(gens, mean_plot, label="Fitness Médio (/ 10M)", color="#3498db", linewidth=1.5, linestyle="--")
        ax1.set_title("Evolução do Fitness por Geração (Escala / 10 Milhões)")
        ax1.set_xlabel("Geração")
        ax1.set_ylabel("Fitness (Pontos / 10 Milhões)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(gens, self.history["best_score"], label="Melhor Pontuação", color="#e74c3c", linewidth=2)
        ax2.plot(gens, self.history["mean_score"], label="Pontuação Média", color="#f39c12", linewidth=1.5, linestyle="--")
        ax2.set_title("Evolução das Maçãs Comidas")
        ax2.set_xlabel("Geração")
        ax2.set_ylabel("Total de Maçãs")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.checkpoint_dir, "fitness_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

    def save_checkpoint(self, gen: int, population: list[Genome], is_periodic: bool = False, tag: str = None):
        pop_weights = np.stack([ind.weights for ind in population])
        pop_fitness = np.array([ind.fitness for ind in population], dtype=np.float32)

        best_w = self.best_overall_genome.weights if self.best_overall_genome is not None else pop_weights[0]

        checkpoint_data = {
            "generation": gen + 1,
            "population_weights": pop_weights,
            "fitnesses": pop_fitness,
            "best_overall_fitness": float(self.best_overall_fitness),
            "best_overall_weights": best_w,
            "history_best_fitness": np.array(self.history["best_fitness"], dtype=np.float32),
            "history_mean_fitness": np.array(self.history["mean_fitness"], dtype=np.float32),
            "history_best_score": np.array(self.history["best_score"], dtype=np.int32),
            "history_mean_score": np.array(self.history["mean_score"], dtype=np.float32),
        }

        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.npz")
        np.savez_compressed(latest_path, **checkpoint_data)

        if tag:
            tag_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.npz")
            np.savez_compressed(tag_path, **checkpoint_data)

        if is_periodic:
            periodic_path = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{gen + 1:04d}.npz")
            np.savez_compressed(periodic_path, **checkpoint_data)
            best_in_gen = max(population, key=lambda ind: ind.fitness)
            best_gen_path = os.path.join(self.checkpoint_dir, f"best_gen_{gen + 1:04d}.npy")
            best_in_gen.save(best_gen_path)

        self.plot_history()
        stats_path = os.path.join(self.checkpoint_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def load_checkpoint(self, resume_target: str) -> tuple[list[Genome], int]:
        target_str = str(resume_target).strip().lower()
        filepath = None

        if target_str in ("latest", "auto", "none", "true", ""):
            candidate = os.path.join(self.checkpoint_dir, "checkpoint_latest.npz")
            if os.path.exists(candidate):
                filepath = candidate
            else:
                raise FileNotFoundError(f"Checkpoint mais recente não encontrado em '{candidate}'.")
        elif target_str.isdigit():
            gen_num = int(target_str)
            cand1 = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{gen_num:04d}.npz")
            cand2 = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{gen_num}.npz")
            if os.path.exists(cand1):
                filepath = cand1
            elif os.path.exists(cand2):
                filepath = cand2
            else:
                raise FileNotFoundError(
                    f"Checkpoint para a geração {gen_num} não encontrado em '{self.checkpoint_dir}'. "
                    f"Certifique-se de que o arquivo 'checkpoint_gen_{gen_num:04d}.npz' existe."
                )
        else:
            if os.path.exists(resume_target):
                filepath = resume_target
            elif os.path.exists(os.path.join(self.checkpoint_dir, resume_target)):
                filepath = os.path.join(self.checkpoint_dir, resume_target)
            else:
                raise FileNotFoundError(f"Arquivo de checkpoint '{resume_target}' não encontrado.")

        print(f"[CHECKPOINT] Carregando estado a partir de: {filepath}")

        if filepath.endswith(".npy"):
            base_genome = Genome.load(filepath)
            print(f"[CHECKPOINT] Populando a partir do genoma único '{filepath}'...")
            population = [base_genome.clone()]
            for _ in range(self.population_size - 1):
                mut_w = self.ga.mutate(base_genome.weights, mutation_rate=0.08, sigma=0.15)
                population.append(Genome(weights=mut_w))
            self.best_overall_genome = base_genome.clone()
            self.best_overall_fitness = 0.0
            return population, 0

        data = np.load(filepath, allow_pickle=True)
        pop_weights = data["population_weights"]
        fitnesses = data["fitnesses"]
        start_gen = int(data["generation"])

        if "history_best_fitness" in data:
            bf = data["history_best_fitness"].tolist()
            mf = data["history_mean_fitness"].tolist()
            self.history["best_score"] = data["history_best_score"].tolist()
            self.history["mean_score"] = data["history_mean_score"].tolist()

            if len(bf) > 0 and max(bf) > 1000.0 and max(bf[:min(100, len(bf))]) < 10.0:
                bf = [x * 100000.0 if x < 1000.0 else x for x in bf]
                mf = [x * 100000.0 if x < 1000.0 else x for x in mf]

            self.history["best_fitness"] = bf
            self.history["mean_fitness"] = mf

        if "best_overall_fitness" in data:
            raw_fit = float(data["best_overall_fitness"])
            if raw_fit < 10.0 and len(self.history["best_fitness"]) > 0 and max(self.history["best_fitness"]) > 1000.0:
                raw_fit *= 100000.0
            self.best_overall_fitness = raw_fit
        if "best_overall_weights" in data and data["best_overall_weights"].size > 0:
            self.best_overall_genome = Genome(weights=data["best_overall_weights"], fitness=self.best_overall_fitness)

        fixed_fitnesses = []
        for f in fitnesses:
            val = float(f)
            if val < 10.0 and self.best_overall_fitness > 1000.0:
                val *= 100000.0
            fixed_fitnesses.append(val)

        population = [Genome(weights=w, fitness=f) for w, f in zip(pop_weights, fixed_fitnesses)]

        if len(population) != self.population_size:
            print(f"[AVISO] População no checkpoint ({len(population)}) diferente de --pop ({self.population_size}).")
            if self.population_size > len(population):
                diff = self.population_size - len(population)
                elites = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:max(1, len(population) // 10)]
                for i in range(diff):
                    parent = elites[i % len(elites)]
                    mut_w = self.ga.mutate(parent.weights, mutation_rate=0.05, sigma=0.15)
                    population.append(Genome(weights=mut_w))
            else:
                population.sort(key=lambda ind: ind.fitness, reverse=True)
                population = population[:self.population_size]

        print(f"[OK] Checkpoint carregado com sucesso!")
        print(f"     Geração de Retomada: {start_gen + 1}")
        print(f"     Cobras na População: {len(population)}")
        print(f"     Melhor Fitness Histórico: {self.best_overall_fitness:.1f}")

        return population, start_gen

    def train(self):
        device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else f"CPU ({self.num_workers} núcleos)"
        print("=" * 70)
        print(f"INICIANDO TREINAMENTO - ALGORITMO GENÉTICO SNAKE")
        print(f"Dispositivo de Execução: {device_name.upper()}")
        if self.infinite:
            print(f"População: {self.population_size} | Modo: CONTÍNUO (Indefinido - pare com Ctrl+C)")
        else:
            print(f"População: {self.population_size} | Gerações Alvo: {self.generations}")
        print(f"Grids de Avaliação: {self.grid_sizes} | Salvamento a cada: {self.save_every} gerações")
        print("=" * 70)

        if self.resume_target is not None:
            population, start_gen = self.load_checkpoint(self.resume_target)
            if not self.infinite and start_gen >= self.generations:
                old_target = self.generations
                self.generations = start_gen + 50
                print(f"[AVISO] Geração do checkpoint ({start_gen}) >= gerações alvo ({old_target}).")
                print(f"        Meta estendida automaticamente para {self.generations} gerações.")
        else:
            population = self.ga.init_population()
            start_gen = 0

        pool = None
        if self.device != "cuda":
            pool = mp.Pool(processes=self.num_workers)

        start_time = time.time()
        completed_in_session = 0
        recent_durations = []
        last_completed_gen = start_gen - 1

        try:
            for gen in range(start_gen, self.generations):
                gen_start = time.time()
                if self.infinite:
                    print(f"\n>>> GERAÇÃO {gen + 1:04d} [CONTÍNUO] (Avaliando {self.population_size} cobras...)", flush=True)
                else:
                    print(f"\n>>> GERAÇÃO {gen + 1:03d}/{self.generations:03d} (Avaliando {self.population_size} cobras...)", flush=True)

                if self.device == "cuda":
                    results = self.gpu_evaluator.evaluate_population(population, self.grid_sizes, verbose=True, trials=self.trials)
                    scores = [r[1] for r in results]
                else:
                    tasks = [(ind.weights, self.grid_sizes, self.trials) for ind in population]
                    results = pool.map(_evaluate_worker, tasks)
                    scores = []
                    for ind, (fitness, score, _) in zip(population, results):
                        ind.fitness = fitness
                        scores.append(score)

                population, stats = self.ga.evolve_population(
                    population,
                    current_gen=gen,
                    max_gens=self.generations
                )

                best_fit = stats["best_fitness"]
                mean_fit = stats["mean_fitness"]
                best_score = max(scores)
                mean_score = float(np.mean(scores))

                self.history["best_fitness"].append(best_fit)
                self.history["mean_fitness"].append(mean_fit)
                self.history["best_score"].append(best_score)
                self.history["mean_score"].append(mean_score)

                record_alert = ""
                if best_fit > self.best_overall_fitness:
                    self.best_overall_fitness = best_fit
                    self.best_overall_genome = stats["best_genome"]
                    best_path = os.path.join(self.checkpoint_dir, "best_overall.npy")
                    self.best_overall_genome.save(best_path)
                    record_alert = f" [NOVO RECORDE: {best_fit:.1f}]"

                completed_in_session += 1
                last_completed_gen = gen

                gen_duration = time.time() - gen_start
                recent_durations.append(gen_duration)
                if len(recent_durations) > 5:
                    recent_durations.pop(0)

                recent_avg_time = sum(recent_durations) / len(recent_durations)

                if self.infinite:
                    elapsed_total = time.time() - start_time
                    if elapsed_total < 3600:
                        elapsed_str = f"{int(elapsed_total // 60)}m {int(elapsed_total % 60):02d}s"
                    else:
                        elapsed_str = f"{int(elapsed_total // 3600)}h {int((elapsed_total % 3600) // 60):02d}m"
                    time_info = f"Decorrido: {elapsed_str}"
                    gen_tag = f"Geracao {gen + 1:04d}"
                else:
                    remaining_gens = self.generations - (gen + 1)
                    eta_seconds = remaining_gens * recent_avg_time
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.0f}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60):02d}s"
                    else:
                        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60):02d}m"
                    time_info = f"Restante: {eta_str}"
                    gen_tag = f"Geracao {gen + 1:03d}/{self.generations:03d}"

                print(
                    f"[OK] {gen_tag} | "
                    f"Tempo: {gen_duration:4.1f}s | "
                    f"Media: {recent_avg_time:4.1f}s/gen | "
                    f"{time_info} | "
                    f"Melhor Fit: {best_fit:8.1f} | "
                    f"Max Macas: {best_score:2d}{record_alert}",
                    flush=True
                )

                is_periodic = ((gen + 1) % self.save_every == 0) or ((gen + 1) == self.generations)
                self.save_checkpoint(gen, population, is_periodic=is_periodic)

        except KeyboardInterrupt:
            print("\n[AVISO] Treinamento pausado pelo usuário via Ctrl+C!")
            if completed_in_session > 0:
                self.save_checkpoint(last_completed_gen, population, is_periodic=False, tag="interrupted")
                print(f"[OK] Progresso salvo na Geração {last_completed_gen + 1}!")
                print(f"     Checkpoint atualizado em: {os.path.join(self.checkpoint_dir, 'checkpoint_latest.npz')}")
                print(f"     Para continuar a qualquer momento, execute:")
                print(f"     python -m training.train --resume")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        total_elapsed = time.time() - start_time
        print("=" * 70)
        print(f"TREINAMENTO FINALIZADO em {total_elapsed / 60:.1f} minutos!")
        if self.best_overall_genome is not None:
            print(f"Melhor Fitness Histórico: {self.best_overall_fitness:.1f}")
            print(f"Melhor Genoma salvo em: {os.path.join(self.checkpoint_dir, 'best_overall.npy')}")
            print(f"Último Checkpoint salvo em: {os.path.join(self.checkpoint_dir, 'checkpoint_latest.npz')}")
            print(f"Gráfico de evolução salvo em: {os.path.join(self.checkpoint_dir, 'fitness_curve.png')}")
            self.plot_history()
        else:
            print("Nenhuma geração foi finalizada por completo nesta sessão.")
        print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Treinamento com Algoritmo Genético para o Snake.")
    parser.add_argument("--pop", type=int, default=200, help="Tamanho da população (padrão: 200)")
    parser.add_argument("--gens", type=int, default=None, help="Número de gerações alvo (padrão: 50 se omitido; use 0 ou --infinite para treino contínuo)")
    parser.add_argument("--infinite", action="store_true", help="Executa o treino de forma contínua/indefinida (pare quando desejar com Ctrl+C)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Dispositivo (cuda ou cpu)")
    parser.add_argument("--grid", type=int, default=15, help="Tamanho da grade única (padrão: 15 para 15x15)")
    parser.add_argument("--multi-grid", action="store_true", help="Se ativado, avalia em múltiplos tabuleiros (10x10 e 15x15)")
    parser.add_argument("--trials", type=int, default=1, help="Número de partidas independentes por cobra (padrão: 1, use 2 para eliminar sorte)")
    parser.add_argument("--save-every", type=int, default=10, help="Salva checkpoint completo a cada X gerações (padrão: 10)")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Retoma o treino a partir do último checkpoint (ou arquivo/geração específica)")
    parser.add_argument("--from-gen", type=int, default=None, help="Atalho para continuar a partir de uma geração específica (ex: --from-gen 50)")
    parser.add_argument("--mutation-rate", type=float, default=0.035, help="Taxa de mutação por gene (padrão: 0.035, ou 3.5%%)")
    parser.add_argument("--mutation-sigma", type=float, default=0.15, help="Intensidade do ruído gaussiano (padrão: 0.15)")
    parser.add_argument("--decay-mutation", action="store_true", help="Se ativado, reduz a mutação ao longo do treino em vez de mantê-la fixa")

    args = parser.parse_args()

    if args.multi_grid:
        grid_sizes = ((10, 10), (15, 15))
    else:
        grid_sizes = ((args.grid, args.grid),)

    is_infinite = args.infinite or (args.gens == 0)
    target_gens = 0 if is_infinite else (50 if args.gens is None else args.gens)

    trainer = Trainer(
        population_size=args.pop,
        generations=target_gens,
        device=args.device,
        grid_sizes=grid_sizes,
        trials=args.trials,
        save_every=args.save_every,
        resume=args.resume,
        from_gen=args.from_gen,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        fixed_mutation=not args.decay_mutation,
        infinite=is_infinite
    )
    trainer.train()
