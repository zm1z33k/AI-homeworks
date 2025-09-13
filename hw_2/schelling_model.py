import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the grid size, empty ratio, similarity threshold, and number of iterations
grid_size = 100  
empty_ratio = 0.1  
similarity_threshold = 0.7  
iterations = 200

# Create the initial grid
num_cells = grid_size * grid_size
num_empty = int(empty_ratio * num_cells)
num_agents = num_cells - num_empty
num_type1 = num_agents // 2
num_type2 = num_agents - num_type1

# Shuffle the cells
cells = np.array([1] * num_type1 + [2] * num_type2 + [0] * num_empty)
np.random.shuffle(cells)
grid = cells.reshape((grid_size, grid_size))

# Define the function to get the neighbors of a cell
def get_neighbors(x, y, grid):
    neighbors = []

    # Iterate over the neighbors
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:

            # Skip the cell itself
            if dx == 0 and dy == 0:
                continue

            # Get the neighbor coordinates
            nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
            neighbors.append(grid[nx, ny])
    return neighbors

# Define the function to check if a cell is happy
def is_happy(x, y, grid):
    agent = grid[x, y]
    if agent == 0:
        return True 
    
    # Get the neighbors of the cell
    neighbors = get_neighbors(x, y, grid)
    same_type = sum(1 for n in neighbors if n == agent)
    total_neighbors = sum(1 for n in neighbors if n != 0)
    
    # Return True if the agent is happy, False otherwise
    return total_neighbors == 0 or (same_type / total_neighbors) >= similarity_threshold

# Define the function to move unhappy agents
def move_unhappy_agents(grid):

    # Shuffle the empty positions
    empty_positions = list(zip(*np.where(grid == 0)))
    np.random.shuffle(empty_positions)
    
    unhappy_agents = []
    happy_count = 0

    # Iterate over all cells
    for x in range(grid_size):
        for y in range(grid_size):
            if grid[x, y] != 0:
                if is_happy(x, y, grid):
                    happy_count += 1
                else:
                    unhappy_agents.append((x, y))
    
    np.random.shuffle(unhappy_agents)

    # Move the unhappy agents to empty positions
    for (x, y), (new_x, new_y) in zip(unhappy_agents, empty_positions):
        grid[new_x, new_y] = grid[x, y]
        grid[x, y] = 0
    
    # Return the number of unhappy agents and happy agents
    return len(unhappy_agents), happy_count

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
cmap = plt.get_cmap("Spectral", 3)

# Initialize the grid plot
im = ax1.imshow(grid, cmap=cmap)
ax1.set_title("Schellingův model segregace")

# Initialize the line plot
unhappy_counts = []
happy_counts = []
line1, = ax2.plot([], [], label="Počet nespokojených agentů", color='red')
line2, = ax2.plot([], [], label="Počet spokojených agentů", color='green')
ax2.set_xlim(0, iterations)
ax2.set_ylim(0, grid_size * grid_size)
ax2.set_xlabel("Iterace")
ax2.set_ylabel("Počet agentů")
ax2.set_title("Vývoj spokojenosti v čase")
ax2.legend()

# Update function for the animation
def update(frame):
    global grid
    if frame < iterations:
        unhappy_count, happy_count = move_unhappy_agents(grid)
        unhappy_counts.append(unhappy_count)
        happy_counts.append(happy_count)
        im.set_array(grid)
        line1.set_data(range(len(unhappy_counts)), unhappy_counts)
        line2.set_data(range(len(happy_counts)), happy_counts)
        return im, line1, line2
    else:
        return im, line1, line2

# Create the animation
ani = FuncAnimation(fig, update, frames=iterations + 1, interval=200, repeat=False)

# Show the plot
plt.tight_layout()
plt.show()
