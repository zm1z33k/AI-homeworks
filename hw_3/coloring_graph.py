from datetime import datetime
import time
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt

# Redirect print statements to both console and a log file
class Logger:
    def __init__(self, log_dir=None):
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        self.log_path = os.path.join(log_dir, log_filename)
        self.terminal = sys.stdout
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    # Redirect print statements to both console and log file
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    # Flush the output buffers
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

sys.stdout = Logger()

# Vytvoření barevného formátu pro výpis
time_format = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
info = time_format + " [\033[92mINFO\033[0m]"
warning = time_format + " [\033[93mWARNING\033[0m]"
item = time_format + " [\033[94mITEM\033[0m]"

# Načtení grafu ve formátu DIMACS
def read_dimacs(filename):
    
    # Načte graf ve formátu DIMACS z textového souboru.  
    print(f"{info} Čtu soubor: {filename}")
    graph = {}

    # Načtení souboru řádek po řádku
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Ignorování komentářů a prázdných řádků
            if not line or line.startswith("c"):
                continue

            # Načtení počtu vrcholů
            if line.startswith("p"):
                parts = line.split()
                n = int(parts[2])
                print(f"{info} Nalezeno {n} vrcholů.")

                # Inicializace všech vrcholů
                for i in range(1, n+1):
                    graph[i] = set()

            # Načtení hran
            elif line.startswith("e"):
                parts = line.split()
                u = int(parts[1])
                v = int(parts[2])

                # Přidání hrany mezi vrcholy (graf je neorientovaný)
                if u not in graph:
                    graph[u] = set()
                if v not in graph:
                    graph[v] = set()
                graph[u].add(v)
                graph[v].add(u)
    print(f"{info} Dokončeno načítání grafu.")
    return graph

# Obarvení grafu pomocí DSATUR algoritmu
def dsatur_coloring(graph):
   
    # Spouští DSATUR algoritmus pro obarvení grafu.
    print(f"{info} Spouštím DSATUR obarvení grafu.")
    coloring = {}

    # Inicializace stupně a saturace vrcholů
    degrees = {v: len(neighbors) for v, neighbors in graph.items()}
    saturation = {v: 0 for v in graph}
    
    # Hlavní smyčka pro obarvení všech vrcholů
    while len(coloring) < len(graph):
        candidate = None
        max_sat = -1
        max_deg = -1

        # Výběr neobarveného vrcholu s nejvyšší saturací a při shodě s vyšším stupněm
        for v in graph:
            if v not in coloring:
                sat = saturation[v]
                deg = degrees[v]

                # Výběr vrcholu s nejvyšší saturací a při shodě s vyšším stupněm
                if sat > max_sat or (sat == max_sat and deg > max_deg):
                    max_sat = sat
                    max_deg = deg
                    candidate = v
        v = candidate

        # Přiřazení nejmenší možné barvy
        used_colors = set(coloring.get(neigh) for neigh in graph[v] if neigh in coloring)
        color = 1

        # Hledání nejmenší možné barvy
        while color in used_colors:
            color += 1
        coloring[v] = color
        print(f"{item} DSATUR: Vrchol {v} přiřazen barva {color}")
        
        # Aktualizace stupně a saturace sousedů
        for neighbor in graph[v]:
            if neighbor not in coloring:
                neighbor_used = set(coloring.get(u) for u in graph[neighbor] if u in coloring)
                saturation[neighbor] = len(neighbor_used)
    print(f"{info} DSATUR obarvení dokončeno.")
    return coloring

# Lokální vyhledávání pro zlepšení obarvení
def local_search_coloring(graph, initial_coloring):
    print(f"{info} Spouštím lokální vyhledávání pro zlepšení obarvení.")
    coloring = initial_coloring.copy()
    improved = True
    iteration = 0
    max_iterations = 10000

    # Hlavní smyčka pro zlepšení obarvení
    while improved and iteration < max_iterations:
        
        improved = False
        iteration += 1

        # Pro každý vrchol se zkusí změnit barvu na nižší možnou
        for v in graph:
            current_color = coloring[v]

            # Zkoušení všech možných barev od 1 do aktuálního maxima
            for new_color in range(1, max(coloring.values()) + 1):
                if new_color == current_color:
                    continue

                # Kontrola, zda je nová barva platná
                conflict = False

                # Kontrola konfliktu s obarvením sousedů
                for neigh in graph[v]:

                    # Pokud je soused obarvený stejnou barvou, je konflikt
                    if coloring.get(neigh, 0) == new_color:
                        conflict = True
                        break

                # Pokud není konflikt, změníme barvu
                if not conflict:
                    print(f"{item} Vrchol {v}: barva změněna z {current_color} na {new_color}")
                    coloring[v] = new_color
                    improved = True
                    break

        # Snížení počtu barev, pokud je to možné
        used_colors = set(coloring.values())
        max_color = max(used_colors)

        # Zkoušení snížení barev
        for color in range(1, max_color + 1):

            # Pokud je barva použita, zkusíme ji snížit
            if color not in used_colors:

                # Přemapování vyšších barev na nižší
                for v in coloring:

                    # Pokud je barva vyšší než aktuální, snížíme ji
                    if coloring[v] > color:
                        coloring[v] -= 1

                # Kontrola, zda se snížil počet barev
                used_colors = set(coloring.values())
                max_color = max(used_colors)
                improved = True
                print(f"{info} Snížen počet barev na {max_color}.")
                break

    # Informace o ukončení lokálního vyhledávání                
    if iteration >= max_iterations:
        print(f"{warning} Lokální vyhledávání bylo ukončeno po dosažení maximálního počtu iterací.")
    else:
        print(f"{info} Lokální vyhledávání dokončeno.")
    print(f"{info} Lokální vyhledávání dokončeno po {iteration} iteracích.")
    return coloring

# Vizualizace grafu s obarvením
def visualize_graph(graph, coloring, best_k, elapsed_time, algo_name):

    # Vytvoří vizualizaci grafu s obarvenými vrcholy podle poskytnutého řešení.
    # Přidá titulek s informacemi o nejlepší použité barvě (k) a době běhu.
    G = nx.Graph()
    for node, neighbors in graph.items():
        G.add_node(node)
        for neigh in neighbors:
            # Každou hranu přidáme jen jednou
            if node < neigh:
                G.add_edge(node, neigh)
                
    pos = nx.spring_layout(G)
    # Přiřadíme každému vrcholu barvu podle obarvení (používáme colormap)
    node_colors = []
    cmap = plt.get_cmap('tab20')

    # Přidání barvy pro každý vrchol
    for node in G.nodes():
        color_num = coloring[node]

        # Normalizace barvy (index od 0 do best_k-1)
        node_colors.append(cmap((color_num - 1) / best_k))
    
    # Vykreslení grafu s obarvením
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', font_weight='bold')
    plt.title(f"Best Coloring: k = {best_k} by {algo_name} | Time: {elapsed_time:.2f} sec")
    plt.show()

# Hlavní funkce pro spuštění obarvování grafu
def main():

    # Cesta k souboru s grafem ve formátu DIMACS
    filename = "D:\\Unicorn Collage\\4.semestr\\AI_new\\hw_3\\dsjc125.9.col.txt"  # Upravte cestu dle potřeby
    print(f"{info} Program spuštěn.")
    
    start_time = time.time()
    
    # Načtení grafu
    graph = read_dimacs(filename)
    print(f"{info} Graf byl úspěšně načten.")
    
    # 1. Obarvení pomocí DSATUR algoritmu
    dsatur_solution = dsatur_coloring(graph)
    dsatur_k = max(dsatur_solution.values())
    print(f"{info} DSATUR řešení použilo {dsatur_k} barev.")
    
    # 2. Lokální vyhledávání na základě DSATUR řešení
    local_solution = local_search_coloring(graph, dsatur_solution)
    local_k = max(local_solution.values())
    print(f"{info} Lokální vyhledávání (DSATUR) dosáhlo obarvení s {local_k} barvami.")
    
    # Vyhodnocení nejlepšího řešení (minimální počet barev)
    best_k = min(dsatur_k, local_k)
    if dsatur_k == best_k:
        best_solution = dsatur_solution
        best_algo = "DSATUR"
    elif local_k == best_k:
        best_solution = local_solution
        best_algo = "Local Search"
    else:
        best_solution = dsatur_solution
        best_algo = "DSATUR"
    
    elapsed_time = time.time() - start_time
    
    print(f"{info} Běh programu trval {elapsed_time:.2f} sekund.")
    print(f"\n{item} Finální obarvení vrcholů (DSATUR):")

    # Výpis obarvení vrcholů
    for v in sorted(dsatur_solution.keys()):
        print(f"{item} Vrchol {v}: barva {dsatur_solution[v]}")
    
    # Vizualizace grafu s nejlepší nalezenou barvou a celkovým časem běhu
    visualize_graph(graph, best_solution, best_k, elapsed_time, best_algo)

if __name__ == "__main__":
    main()