import heapq
from collections import deque
import pygame
import time
import random
import csv

# Inicializace Pygame
pygame.init()

# Velikost jednoho dlaždice
TILESIZE = 32

# Velikost mapy
MAP_WIDTH = 24
MAP_HEIGHT = 16

# Vytvoření okna
WIN = pygame.display.set_mode((TILESIZE * MAP_WIDTH, TILESIZE * MAP_HEIGHT))
pygame.display.set_caption("Pathfinding Visualizer")

# Načtení obrázků a jejich škálování
TILE = pygame.transform.scale(pygame.image.load("hw_8\\tile.jpg"), (TILESIZE, TILESIZE))
TREE1 = pygame.transform.scale(pygame.image.load("hw_8\\tree1.jpg"), (TILESIZE, TILESIZE))
TREE2 = pygame.transform.scale(pygame.image.load("hw_8\\tree2.jpg"), (TILESIZE, TILESIZE))
HOUSE1 = pygame.transform.scale(pygame.image.load("hw_8\\house1.jpg"), (TILESIZE, TILESIZE))
HOUSE2 = pygame.transform.scale(pygame.image.load("hw_8\\house2.jpg"), (TILESIZE, TILESIZE))
HOUSE3 = pygame.transform.scale(pygame.image.load("hw_8\\house3.jpg"), (TILESIZE, TILESIZE))
FLAG = pygame.transform.scale(pygame.image.load("hw_8\\flag.jpg"), (TILESIZE, TILESIZE))
MTILE = pygame.transform.scale(pygame.image.load("hw_8\\markedtile.jpg"), (TILESIZE, TILESIZE))
UFO = pygame.transform.scale(pygame.image.load("hw_8\\ufo.jpg"), (TILESIZE, TILESIZE))

LEVEL_FONT = pygame.font.SysFont("arial", 24)

# Funkce pro výpočet Manhattanovy vzdálenosti
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Funkce pro rekonstrukci cesty z mapy předchozích uzlů
def reconstruct_path(came_from, current):
    path = deque()
    while current in came_from:
        path.appendleft(current)
        current = came_from[current]
    path.appendleft(current)
    return path

# Třída pro prostředí, které obsahuje mapu a metody pro plánování cesty
class Env:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.startx = 0
        self.starty = 0
        self.goalx = width - 1
        self.goaly = height - 1
        self.grid = [[0 for _ in range(height)] for _ in range(width)]

    # Nastavení mapy prostředí
    def set_map(self, grid):
        self.grid = [row[:] for row in grid]

    # Generování překážek v prostředí
    def generate_obstacles(self):
        self.grid = [[0 for _ in range(self.height)] for _ in range(self.width)]
        path = set()
        x, y = self.startx, self.starty

        # Vytvoření cesty od startu k cíli
        while x != self.goalx or y != self.goaly:
            path.add((x, y))
            if x < self.goalx:
                x += 1
            elif y < self.goaly:
                y += 1
        path.add((self.goalx, self.goaly))

        # Nastavení překážek na náhodných místech
        for i in range(self.width):
            for j in range(self.height):
                if (i, j) not in path and random.random() < 0.2:
                    self.grid[i][j] = random.choice([1, 2, 3, 4, 5])

    # Nastavení startovní a cílové pozice
    def set_start(self, x, y):
        self.startx, self.starty = x, y

    # Nastavení cílové pozice
    def set_goal(self, x, y):
        self.goalx, self.goaly = x, y

    # Získání typu dlaždice na dané pozici
    def get_tile_type(self, x, y):
        return self.grid[x][y]

    # Získání sousedních dlaždic, které jsou průchozí
    def get_neighbors(self, x, y):
        neighbors = []

        # Kontrola čtyř směrů (nahoru, doprava, dolů, doleva)
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x + dx, y + dy

            # Kontrola, zda jsou souřadnice v platném rozsahu a zda je dlaždice průchozí
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[nx][ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors

    # Plánování cesty pomocí různých algoritmů
    # Podporované metody: "greedy", "dijkstra", "astar"
    def path_planner(self, method="astar"):
        start = (self.startx, self.starty)
        goal = (self.goalx, self.goaly)
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        cost_so_far = {start: 0}
        expanded = []
        visited_set = set([start])
        MAX_STEPS = 5000
        steps = 0

        # Výběr prioritní funkce na základě zvoleného algoritmu
        while open_set:
            if steps > MAX_STEPS:
                print(f"{method} exceeded step limit without finding a path.")
                break
            steps += 1
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = reconstruct_path(came_from, goal)
                return path, expanded, len(visited_set), len(expanded)
            if current in expanded:
                continue
            expanded.append(current)

            # Procházení sousedních dlaždic
            for neighbor in self.get_neighbors(*current):
                new_cost = cost_so_far[current] + 1

                # Kontrola, zda je sousední dlaždice již navštívena a zda je nový náklad menší
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current

                    # Greedy metoda používá Manhattanovu vzdálenost
                    if method == "greedy":
                        priority = manhattan(neighbor, goal)

                    # Dijkstra a A* používají náklady    
                    elif method == "dijkstra":
                        priority = new_cost

                    # A* používá kombinaci nákladů a Manhattanovy vzdálenosti    
                    else:
                        priority = new_cost + manhattan(neighbor, goal)
                    heapq.heappush(open_set, (priority, neighbor))
                    visited_set.add(neighbor)
        print(f"No path found using {method}. Visited: {len(visited_set)}, Expanded: {len(expanded)}")
        return deque(), expanded, len(visited_set), len(expanded)

# Třída pro UFO, které bude pohybovat po cestě
class Ufo:

    # Konstruktor třídy UFO
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path = []
        self.tiles = []

    # Nastavení cesty a rozšířených dlaždic
    def set_path(self, path, expanded):
        self.path = list(path)
        self.tiles = list(expanded)

    # Vykonání kroku na cestě, pokud je k dispozici
    def execute_path(self):
        if self.path:
            return self.path.pop(0)
        return self.x, self.y

    # Posunutí UFO na dané souřadnice
    def move(self, x, y):
        self.x = x
        self.y = y

# Funkce pro vykreslení okna s UFO a prostředím
def draw_window(ufo, env, title=""):
    for i in range(env.width):
        for j in range(env.height):
            t = env.get_tile_type(i, j)
            if t == 1:
                WIN.blit(TREE1, (i*TILESIZE, j*TILESIZE))
            elif t == 2:
                WIN.blit(HOUSE1, (i*TILESIZE, j*TILESIZE))
            elif t == 3:
                WIN.blit(HOUSE2, (i*TILESIZE, j*TILESIZE))
            elif t == 4:
                WIN.blit(HOUSE3, (i*TILESIZE, j*TILESIZE))
            elif t == 5:
                WIN.blit(TREE2, (i*TILESIZE, j*TILESIZE))
            else:
                WIN.blit(TILE, (i*TILESIZE, j*TILESIZE))

    # Vykreslení rozšířených dlaždic a cesty UFO
    for (x, y) in ufo.tiles:
        WIN.blit(MTILE, (x*TILESIZE, y*TILESIZE))

    WIN.blit(FLAG, (env.goalx * TILESIZE, env.goaly * TILESIZE))
    WIN.blit(UFO, (ufo.x * TILESIZE, ufo.y * TILESIZE))

    # Vykreslení startovní pozice UFO
    if title:
        label = LEVEL_FONT.render(title, True, (0, 0, 0))
        WIN.blit(label, (10, 10))

    pygame.display.update()

algorithms = ["greedy", "dijkstra", "astar"]
titles = {"greedy": "Greedy Best-First Search", "dijkstra": "Dijkstra", "astar": "A*"}

# Hlavní funkce pro spuštění vizualizace
def main():
    running = True
    ufo = None
    base_env = Env(MAP_WIDTH, MAP_HEIGHT)
    base_env.generate_obstacles()

    # Vytvoření CSV souboru pro uložení metrik
    with open("metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Visited", "Expanded"])

        # Hlavní smyčka pro vizualizaci
        for algo in algorithms:
            env = Env(MAP_WIDTH, MAP_HEIGHT)
            env.set_map(base_env.grid)
            env.set_start(0, 0)
            env.set_goal(MAP_WIDTH - 1, MAP_HEIGHT - 1)
            ufo = Ufo(env.startx, env.starty)
            draw_window(ufo, env)
            print("Press A for %s" % titles[algo])

            waiting = True

            # Čekání na stisknutí mezerníku pro spuštění algoritmu
            while waiting:

                # Zpracování událostí Pygame
                for event in pygame.event.get():

                    # Pokud je stisknuto tlačítko pro ukončení, ukončíme smyčku
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False

                    # Pokud je stisknuto písmeno A, spustíme algoritmus
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            waiting = False

            # Spuštění plánování cesty pomocí zvoleného algoritmu
            path, expanded, visited_count, expanded_count = env.path_planner(method=algo)
            ufo.set_path(path, expanded)
            title_text = f"{titles[algo]} | Visited: {visited_count} | Expanded: {expanded_count}"
            writer.writerow([titles[algo], visited_count, expanded_count])

            # Vykreslení cesty a rozšířených dlaždic
            for step in range(len(expanded)):
                draw_window(ufo, env, title_text)
                time.sleep(0.02)

            # Pohyb UFO po cestě
            while ufo.path:
                ufo.move(*ufo.execute_path())
                draw_window(ufo, env, title_text)
                time.sleep(0.05)

            time.sleep(1.5)

    pygame.quit()

if __name__ == "__main__":
    main()