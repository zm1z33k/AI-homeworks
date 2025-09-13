import random

# Simulates dice rolls to count 10 consecutive sixes in a streak.

def simulate_dice_rolls(iterations, attempts):
    target_streak = 10
    streak_count = 0
    current_streak = 0
    attemptsArr = []

    for _ in range(attempts):

        for _ in range(iterations):
            roll = random.randint(1, 6)
            if roll == 6:
                current_streak += 1
                if current_streak == target_streak:
                    streak_count += 1
                    current_streak = 0  # Reset streak after achieving target
            else:
                current_streak = 0

        attemptsArr.append(streak_count)
    return attemptsArr

if __name__ == "__main__":
    iterations = 20000000  # Adjust this number if needed
    attempts = 10  # Number of attempts
    result = simulate_dice_rolls(iterations, attempts)
    print(f"With {iterations} on one attempt, and {attempts} attempts on run. Number of times 10 sixes in a row occurred: {result}")
    print(f"Average number of times 10 sixes in a row occurred: {sum(result) / len(result)}")

# 5000 with 200mld iterations on average