import random

# Create a function step that creates a new row according to the rules of the game
def step(row):
    # Create a new row that will contain the new state of the cells
    new_row = [0] * len(row)

    # Iterate through all cells except the edge ones
    for i in range(1, len(row) - 1):
        # Count the number of live neighbors
        neighbors_sum = row[i - 1] + row[i + 1]

        # Set the new state of the cell according to the rules of the game
        if row[i] == 1:  # Live cell
            if neighbors_sum == 0 or neighbors_sum == 2:
                new_row[i] = 1
            else:
                new_row[i] = 0
        else:  # Dead cell
            if neighbors_sum >= 1:
                new_row[i] = 1
            else:
                new_row[i] = 0
    return new_row

# Function to print the row with colors
# Red color for dead cells, green for live cells
RED_BG = "\033[41m"  # Red background
GREEN_BG = "\033[42m"  # Green background

# Function to print the row with colors
def print_row(row):
    for num in row:
        if num == 1:
            print("", f"{GREEN_BG} {num}", end="")           
        else:
            print("", f"{RED_BG} {num}", end="") 
    print("\033[0m")  # Reset to default background


# Initial state (can be changed for experimentation)
# Generate a random initial state
initial_row = [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]

generations = 20
current_row = initial_row

# Print all generations
for j in range(generations):
    print_row(current_row)
    current_row = step(current_row)