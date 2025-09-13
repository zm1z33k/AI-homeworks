import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deap import base, creator, tools

# Terrain generation
def find_segments(terrain, threshold, above=True):
    segments = []
    i = 0

    # Finding segments
    while i < len(terrain):
        if (terrain[i] >= threshold) if above else (terrain[i] < threshold):
            start = i

            # Finding end of segment
            while i < len(terrain) and ((terrain[i] >= threshold) if above else (terrain[i] < threshold)):
                i += 1
            end = i - 1
            segments.append((start, end))
        else:
            i += 1
    return segments

# Finding mountains
def find_mountains(terrain, sea_level=0.5):
    return find_segments(terrain, sea_level, above=True)

# Finding lakes
def find_lakes(terrain, sea_level=0.5):
    return find_segments(terrain, sea_level, above=False)

# Compute terrain metrics
def compute_terrain_metrics(terrain, sea_level=0.5):

    # Finding mountains and lakes
    num_vertices = len(terrain)
    mountains = find_mountains(terrain, sea_level)
    lakes = find_lakes(terrain, sea_level)

    # Computing metrics
    metrics = {
        'num_vertices': num_vertices,
        'num_mountains': len(mountains),
        'num_lakes': len(lakes),
        'lake_sizes': [end - start + 1 for start, end in lakes],
        'variability': np.std(terrain),
        'flooded_percentage': (
            sum(max(0, sea_level - h) for h in terrain) / (sea_level * num_vertices) * 100
        )
    }
    return metrics

# Evaluation function
def eval_terrain(individual):

    # Compute terrain metrics
    metrics = compute_terrain_metrics(individual)
    target_flooded = 30
    avg_lake_size = np.mean(metrics['lake_sizes']) if metrics['lake_sizes'] else 0

    # Fitness function
    fitness = (
        metrics['variability'] * 50 +  # Reduced weight for variability
        metrics['num_lakes'] * 20 +   # Increased weight for lakes
        avg_lake_size * 10 -          # Increased weight for lake size
        abs(metrics['flooded_percentage'] - target_flooded)
    )
    return (fitness,)

# Plotting terrain
def plot_terrain(ax, terrain, metrics, best_gen_text, sea_level=0.5):

    x = range(len(terrain))
    sea = [sea_level] * len(terrain)
    ax.clear()
    ax.fill_between(x, sea, color="lightblue")
    ax.fill_between(x, terrain, color="brown")

    # Plotting mountains
    textstr = (
        f"Pocet vrcholu: {metrics['num_vertices']}\n"
        f"Pocet hor: {metrics['num_mountains']}\n"
        f"Pocet jezer: {metrics['num_lakes']}\n"
        f"Variabilita: {metrics['variability']:.2f}\n"
        f"Procento zaplavenych: {metrics['flooded_percentage']:.2f}\n"
        f"Velikosti jezer: {metrics['lake_sizes']}\n"
        f"{best_gen_text}"
    )

    # Annotating text
    ax.text(
        0.02, 0.95, textstr,
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(terrain) - 1)
    ax.axis("off")

# DEAP setup
def setup_deap():

    # Creating DEAP classes
    creator.create("FitnessMax", base.Fitness, weights=(0.5,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: random.uniform(0.5, 0.6))  # Flatter initial terrain
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 20)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_terrain)
    toolbox.register("mate", tools.cxTwoPoint)
    
    # Mutation with Gaussian noise and smaller mutation probability
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.3)  # Smaller mutations
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Evolution function
def evolve_population(toolbox, population, num_generations, cxpb, mutpb):
    best_individuals_per_gen = []
    best_fitnesses = []
    global_best_fitness = -float("inf")
    best_gen_index = 0

    # Evolution loop
    for gen in range(num_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # Crossover
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        # Assign fitness
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring
        best = tools.selBest(population, 1)[0]
        current_fitness = best.fitness.values[0]
        best_individuals_per_gen.append(list(best))
        best_fitnesses.append(current_fitness)

        # Update best fitness
        if current_fitness > global_best_fitness:
            global_best_fitness = current_fitness
            best_gen_index = gen
        print(f"Generace {gen} Nejlepsi fitness: {current_fitness:.2f}")
    return best_individuals_per_gen, best_fitnesses, global_best_fitness, best_gen_index

# Main function
def main():
    random.seed(42)
    toolbox = setup_deap()

    # Testing terrain metrics
    population_size = 100
    num_generations = 50
    cxpb = 0.5
    mutpb = 0.2

    # Initial population
    population = toolbox.population(n=population_size)
    fitnesses = list(map(toolbox.evaluate, population))

    # Assign fitness
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    best_individuals_per_gen, _, global_best_fitness, best_gen_index = evolve_population(
        toolbox, population, num_generations, cxpb, mutpb
    )

    # Plotting best individual
    best_ind = tools.selBest(population, 1)[0]
    metrics = compute_terrain_metrics(best_ind)
    print("Nejlepsi jedinec:", best_ind)
    print("Metriky:", metrics)
    best_gen_text = f"Nejlepsi generace {best_gen_index} s fitness {global_best_fitness:.2f}"
    frame_list = []
    hold_frames = 3

    # Animation
    for i in range(len(best_individuals_per_gen)):
        frame_list.append(i)

        # Hold best individual
        if i == best_gen_index:
            frame_list.extend([i] * hold_frames)
    fig, ax = plt.subplots()

    # Update function
    def update(frame):
        terrain = best_individuals_per_gen[frame]
        m = compute_terrain_metrics(terrain)
        gen_text = f"Generace {frame} Nejlepsi generace {best_gen_index} s fitness {global_best_fitness:.2f}"
        plot_terrain(ax, terrain, m, gen_text)
        ax.set_title(f"Generace {frame}", fontsize=10)
        return []
    ani = FuncAnimation(fig, update, frames=frame_list, interval=1000, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
