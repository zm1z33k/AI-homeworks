import random
from deap import base, creator, tools, algorithms

# Definice délky genomu (2^n pro n předchozích kol historie)
HISTORY_LENGTH = 3  # Počet předchozích kol, která se berou v úvahu
GENOME_LENGTH = 2 ** (2 * HISTORY_LENGTH)  # Všechny možné kombinace historie
'''
def zrada(moje_historie, protihracova_historie):
    genome = [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
   
    # Zajištění, že historie má správnou délku
    moje_historie = [0] * (3 - len(moje_historie)) + moje_historie[-3:]
   
    # Zajištění, že historie protihráče má správnou délku
    protihracova_historie = [0] * (3 - len(protihracova_historie)) + protihracova_historie[-3:]
   
    index = 0
 
    # Vytvoření indexu z historie
    for i in range(3):
        index = (index << 1) | moje_historie[i]
        index = (index << 1) | protihracova_historie[i]
   
    return genome[index]
'''
# Dekódování genomu pro určení dalšího tahu
def zrada(moje_historie, protihracova_historie, genome):
    
    # Zajištění, že historie je prázdná, pokud není zadána
    if genome is None:

        # Vytvoření prázdného genomu, pokud není zadán
        genome = [0] * GENOME_LENGTH

    # Zajištění, že historie má správnou délku
    moje_historie = [0] * (HISTORY_LENGTH - len(moje_historie)) + moje_historie[-HISTORY_LENGTH:]
    
    # Zajištění, že historie protihráče má správnou délku
    protihracova_historie = [0] * (HISTORY_LENGTH - len(protihracova_historie)) + protihracova_historie[-HISTORY_LENGTH:]
   
    index = 0

    # Vytvoření indexu na základě historie
    for i in range(HISTORY_LENGTH):

        # Posun indexu doleva a přidání hodnoty z mé historie
        index = (index << 1) | moje_historie[i]

        # Posun indexu doleva a přidání hodnoty z historie protihráče
        index = (index << 1) | protihracova_historie[i]
    
    # Převod indexu na hodnotu v genomu
    return genome[index]

# Fitness funkce pro vyhodnocení genomu
def fitness(individual):
    genome = individual
    opponent_history = []
    my_history = []
    score = 0

    for _ in range(100):  # Simulace 100 kol

        # Náhodná strategie protihráče
        opponent_move = random.randint(0, 1)
        my_move = zrada(my_history, opponent_history, genome)
        
        # Přidání aktuálních tahů do historie
        my_history.append(my_move)
        opponent_history.append(opponent_move)
        
        if my_move == 0 and opponent_move == 0:  # Oba spolupracují
            score += 2
        elif my_move == 0 and opponent_move == 1:  # Já spolupracuji, protihráč zradí
            score += 0
        elif my_move == 1 and opponent_move == 0:  # Já zradím, protihráč spolupracuje
            score += 5
        elif my_move == 1 and opponent_move == 1:  # Oba zradí
            score += 1

    return score,

# Nastavení DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Vytvoření třídy pro jedince s maximální fitness
# a nastavení fitness funkce na maximální hodnotu
creator.create("Individual", list, fitness=creator.FitnessMax)

# Registrace operátorů pro genetický algoritmus
toolbox = base.Toolbox()

# Registrace atributu (genomu) jako náhodného celého čísla 0 nebo 1
toolbox.register("attr_bool", random.randint, 0, 1)

# Registrace jedince jako opakování atributu (genomu) s délkou GENOME_LENGTH
# a nastavení fitness funkce na maximální hodnotu
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=GENOME_LENGTH)

# Registrace populace jako opakování jedince
# a nastavení fitness funkce na maximální hodnotu
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registrace mojí fitness funkce
toolbox.register("evaluate", fitness)

# Registrace cross operátoru (křížení od obou rodičů)
toolbox.register("mate", tools.cxTwoPoint)

# Registrace operátoru mutace (překlopení bitu s pravděpodobností 1 %)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)

# Registrace operátoru selekce (turnajová selekce s velikostí turnaje 3)
toolbox.register("select", tools.selTournament, tournsize=3)

# Hlavní spuštění
if __name__ == "__main__":

    # Vytvoření populace jedinců
    population = toolbox.population(n=100)
    
    # Počet generací pro spuštění genetického algoritmu
    ngen = 50

    # Pravděpodobnost křížení: pravděpodobnost, že si dva jedinci vymění části genomu
    cxpb = 0.5

    # Pravděpodobnost mutace: pravděpodobnost, že se bit v genomu překlopí
    mutpb = 0.2

    # Spuštění genetického algoritmu
    result_population, _ = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    # Získání nejlepšího jedince
    best_individual = tools.selBest(result_population, k=1)[0]
    print("Nejlepší genom:", best_individual)
    print("Fitness genomu:", fitness(best_individual)[0])

    # Příklad použití funkce zrada
    moje_historie = [0, 1, 0, 0]
    protihracova_historie = [1, 0, 1, 0]