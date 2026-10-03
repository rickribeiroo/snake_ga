def compute_fitness(
    steps: int,
    score: int,
    grid_width: int,
    grid_height: int,
    reason: str = "",
    won: bool = False,
    distance_reward: float = 0.0
) -> float:
    grid_area = grid_width * grid_height

    if score == 0:
        base_survival = min(steps, 40) * 0.5
    else:
        step_cap = 2 * grid_area
        base_survival = min(steps, step_cap)

    if score > 0:
        food_score = (score ** 3) * 1000.0
    else:
        food_score = 0.0

    coverage_ratio = (score + 3) / grid_area
    coverage_bonus = coverage_ratio * 500.0

    penalties = 0.0
    bonuses = 0.0

    if won:
        bonuses += 100000.0
    elif reason == "self_collision":
        food_score *= 0.5
        penalties += 300.0 + (score * 20.0)
    elif reason == "starvation":
        penalties += 200.0
    elif reason == "wall_collision":
        penalties += 50.0

    if score == 0 and steps < 8:
        penalties += 50.0

    fitness = base_survival + food_score + coverage_bonus + distance_reward + bonuses - penalties
    scaled_fitness = fitness / 100000.0

    return max(0.001, float(scaled_fitness))
