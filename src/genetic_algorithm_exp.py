import numpy as np
import random
import time
from random import randint, shuffle
import matplotlib.pyplot as plt
from collections import defaultdict

#Single experiment genetic algorithm for keyboard layout optimization

# ============ PRECOMPUTE MATRICES ============
def get_finger_assigned(position):
    """Returns finger assignment for key position"""
    row = position // 10
    col = position % 10
    finger_map = {
        0: ('L', 0, 1), 1: ('L', 1, 2), 2: ('L', 2, 3), 3: ('L', 3, 4), 4: ('L', 3, 4),
        5: ('R', 3, 4), 6: ('R', 3, 4), 7: ('R', 2, 3), 8: ('R', 1, 2), 9: ('R', 0, 1)
    }
    hand, finger, strength = finger_map[col]
    return hand, finger, strength, row, col

def euclidean_distance(pos1, pos2):
    """Euclidean distance between two key positions"""
    _, _, _, row1, col1 = get_finger_assigned(pos1)
    _, _, _, row2, col2 = get_finger_assigned(pos2)
    return np.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)

def finger_penalty(pos1, pos2):
    """Compute penalty based on finger usage"""
    hand1, finger1, strength1, row1, col1 = get_finger_assigned(pos1)
    hand2, finger2, strength2, row2, col2 = get_finger_assigned(pos2)

    penalty = 0
    if hand1 == hand2 and finger1 == finger2 and pos1 != pos2:
        penalty += 1.0
        if strength1 <= 2:
            penalty += 2.0
    elif hand1 == hand2:
        penalty += 1.0
    penalty += -1.0
    
    row_diff = abs(row1 - row2)
    if row_diff == 1:
        penalty += 0.2
        if strength1 <= 2 or strength2 <= 2:
            penalty += 0.15
    elif row_diff == 2:
        penalty += 0.8
        if strength1 <= 2 or strength2 <= 2:
            penalty += 0.5
    
    if finger1 == 0 or finger2 == 0:
        penalty += 0.15
    if finger1 == 1 or finger2 == 1:
        penalty += 0.1
    
    if col1 == col2 and row_diff > 0:
        penalty += 0.3
        if col1 in [0, 9]:
            penalty += 0.2
        elif col1 in [1, 8]:
            penalty += 0.1
    
    return penalty

def precompute_cost_matrices():
    """
    Precompute all position-to-position costs (30x30)
    """

    print("Precomputing cost matrices...")
    distances = np.zeros((30, 30))
    penalties = np.zeros((30, 30))
    
    for i in range(30):
        for j in range(30):
            distances[i, j] = euclidean_distance(i, j)
            penalties[i, j] = finger_penalty(i, j)
    
    print("✓ Cost matrices ready")
    return distances, penalties

# Global matrices (computed once)
DISTANCE_MATRIX, PENALTY_MATRIX = precompute_cost_matrices()

# ============ TEXT PROCESSING ============
def load_text_from_file(filename, sample_size=None):
    """Load and optionally sample text"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read().lower()
        
        if sample_size and sample_size < len(text):
            text = text[:sample_size]
            print(f"Using first {sample_size:,} characters")
        
        return text
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def precompute_bigrams(text):
    """Precompute bigram frequencies"""
    print("Computing bigram frequencies...")
    bigrams = defaultdict(int)
    
    for i in range(len(text) - 1):
        bigram = (text[i], text[i + 1])
        bigrams[bigram] += 1
    
    print(f"✓ Found {len(bigrams):,} unique bigrams from {len(text):,} chars")
    return dict(bigrams)

def fitness_function_optimized(population, bigram_freq):
    """
    OPTIMIZED fitness using precomputed bigrams and cost matrices
    """
    results = []
    
    for keyboard in population:
        # O(1) lookup for character positions
        pos_map = {char: idx for idx, char in enumerate(keyboard)}
        total_cost = 0
        
        for (char1, char2), freq in bigram_freq.items():
            if char1 not in pos_map or char2 not in pos_map:
                continue
            
            pos1 = pos_map[char1]
            pos2 = pos_map[char2]
            
            if pos1 == pos2:
                continue
            
            # Use precomputed matrices (O(1) lookup)
            base_dist = DISTANCE_MATRIX[pos1, pos2]
            finger_cost = PENALTY_MATRIX[pos1, pos2]
            total_multiplier = max(1.0 + finger_cost, 0.1)
            
            total_cost += (base_dist * total_multiplier) * freq
        
        results.append(total_cost)
    
    return results

# ============ GENETIC OPERATORS ============
def init_population(pop_size, known_distributions=False):
    """Initialize population with optional known layouts"""
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
            ',', '.', ';', "'"]
    
    keyboards = []
    
    if known_distributions:
        qwerty = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"]
        ]
        
        dvorak = [
            "'", ',', '.', 'p', 'y', 'f', 'g', 'c', 'r', 'l',  
            'a', 'o', 'e', 'u', 'i', 'd', 'h', 't', 'n', 's',
            ';', 'q', 'j', 'k', 'x', 'b', 'm', 'w', 'v', 'z'   
        ]
        

        qwertz = [
            'q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p',  
            'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',  
            'y', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"   
        ]
        
        
        colemak = [
            'q', 'w', 'f', 'p', 'g', 'j', 'l', 'u', 'y', ';',  
            'a', 'r', 's', 't', 'd', 'h', 'n', 'e', 'i', 'o',  
            'z', 'x', 'c', 'v', 'b', 'k', 'm', ',', '.', "'"   
        ]
            
        
        keyboards = [qwerty, dvorak, qwertz, colemak]
    
    for _ in range(len(keyboards), pop_size):
        keyboards.append(random.sample(letters, len(letters)))
    
    return keyboards

def crossover(parent1, parent2):
    """Two-child crossover"""
    size = len(parent1)
    split_point=random.randint(1, size-1)
    #split_point = random.randint(size // 3, 2 * size // 3)
    
    # Child 1
    child1 = [None] * size
    child1[:split_point] = parent1[:split_point]
    inherited = set(parent1[:split_point])
    p2_filtered = [g for g in parent2 if g not in inherited]
    child1[split_point:] = p2_filtered
    
    # Child 2
    child2 = [None] * size
    child2[:split_point] = parent2[:split_point]
    inherited = set(parent2[:split_point])
    p1_filtered = [g for g in parent1 if g not in inherited]
    child2[split_point:] = p1_filtered
    
    return child1, child2

def crossover_two_point(parent1, parent2):
    """Two-point crossover for permutation encoding"""
    size = len(parent1)
    
    # Seleccionar dos puntos de corte aleatorios (ordenados)
    point1 = random.randint(1, size - 2)
    point2 = random.randint(point1 + 1, size - 1)
    
    # Crear hijo 1
    child1 = [None] * size
    # Copiar segmento central del padre 1
    child1[point1:point2] = parent1[point1:point2]
    # Completar con genes del padre 2 que no estén ya
    inherited1 = set(parent1[point1:point2])
    p2_filtered = [g for g in parent2 if g not in inherited1]
    child1[:point1] = p2_filtered[:point1]
    child1[point2:] = p2_filtered[point1:]
    
    # Crear hijo 2
    child2 = [None] * size
    # Copiar segmento central del padre 2
    child2[point1:point2] = parent2[point1:point2]
    # Completar con genes del padre 1 que no estén ya
    inherited2 = set(parent2[point1:point2])
    p1_filtered = [g for g in parent1 if g not in inherited2]
    child2[:point1] = p1_filtered[:point1]
    child2[point2:] = p1_filtered[point1:]
    
    return child1, child2

def mutation(population, mutation_rate=0.1):
    """Swap mutation"""
    mutated = []
    for layout in population:
        new_layout = layout.copy()
        if np.random.rand() < mutation_rate:
            pos1, pos2 = np.random.choice(30, 2, replace=False)
            new_layout[pos1], new_layout[pos2] = new_layout[pos2], new_layout[pos1]
        mutated.append(new_layout)
    return mutated

def tournament_selection(population_with_fitness, tournament_size=3):
    """Tournament selection"""
    tournament = random.sample(population_with_fitness, k=tournament_size)
    winner = min(tournament, key=lambda item: item[1])
    return winner[0]

# ============ MAIN GA ============
def run_genetic_algorithm(text_file, pop_size=50, generations=50, elite_size=10, 
                        tournament_size=3, crossover_rate=0.8, mutation_rate=0.15, 
                        use_known_distributions=True, seed=None):
    """Optimized genetic algorithm"""
    
    print("=" * 60)
    print("OPTIMIZED GENETIC ALGORITHM FOR KEYBOARD LAYOUT")
    print("=" * 60)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    text = load_text_from_file(text_file)
    if text is None:
        return None, None
    
    bigram_freq = precompute_bigrams(text)
    
    population = init_population(pop_size, use_known_distributions)
    print(f"Population: {pop_size} layouts\n")
    
    history = {"best_fitness": [], "avg_fitness": []}
    best_layout_ever = None
    best_fitness_ever = float('inf')
    
    for gen in range(generations):
        # Usar fitness optimizada
        fitness_scores = fitness_function_optimized(population, bigram_freq)
        
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda item: item[1])
        
        sorted_population = [item[0] for item in population_with_fitness]
        sorted_fitness = [item[1] for item in population_with_fitness]
        
        current_best = sorted_fitness[0]
        current_avg = np.mean(sorted_fitness)
        history["best_fitness"].append(current_best)
        history["avg_fitness"].append(current_avg)
        
        if current_best < best_fitness_ever:
            best_fitness_ever = current_best
            best_layout_ever = sorted_population[0].copy()
        
        print(f"Gen {gen+1:3}/{generations} | Best: {current_best:>12,.1f} | Avg: {current_avg:>12,.1f}")
        
        # Create next generation
        
        next_generation = [layout.copy() for layout in sorted_population[:elite_size]]
        
        num_offspring_needed = pop_size - elite_size
        num_pairs = num_offspring_needed // 2
        offspring = []
        
        for _ in range(num_pairs):
            p1 = tournament_selection(population_with_fitness, tournament_size)
            p2 = tournament_selection(population_with_fitness, tournament_size)
            
            if random.random() < crossover_rate:
                c1, c2 = crossover_two_point(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            offspring.extend([c1, c2])
        
        if len(offspring) < num_offspring_needed:
            p1 = tournament_selection(population_with_fitness, tournament_size)
            p2 = tournament_selection(population_with_fitness, tournament_size)
            c1, _ = crossover(p1, p2)
            offspring.append(c1)
        
        offspring=mutation(offspring[:num_offspring_needed], mutation_rate)
        next_generation.extend(offspring)
        population = next_generation
    
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    return best_layout_ever, history

def print_keyboard(layout, name="Keyboard"):
    """Print keyboard layout"""
    print(f"\n{name}:")
    for row in range(3):
        row_keys = layout[row*10:(row+1)*10]
        print("  " + " ".join(f"{key:>2}" for key in row_keys))

def print_latex_data(history, generations):
    """Print best fitness data in LaTeX format"""
    print("\n" + "="*60)
    print("DATOS PARA LATEX - Copia directamente")
    print("="*60)
    print("\n% Generación y Best Fitness")
    print("% coordinates {(gen, fitness)}")
    print()
    
    for gen in range(1, generations + 1):
        print(f"({gen}, {history['best_fitness'][gen-1]:.1f})")
    
    print("\n" + "="*60)

# ============ MAIN ============
if __name__ == "__main__":
    POPULATION_SIZE = 100000                 # Población grande para mayor diversidad
    GENERATIONS = 100                       # Más generaciones para convergencia
    ELITE_SIZE = 15000                        # 15% de elite (mantiene mejores soluciones)
    TOURNAMENT_SIZE = 5                     # Torneos más competitivos
    CROSSOVER_RATE = 0.85                   # Alto cruce para exploración
    MUTATION_RATE = 0.15                    # Mutación moderada (evita destruir buenos genes)
    SEED=123
    LATEX_FORMAT=True
    CODIFICATION_FORMAT=True
    TEXT_FILE = 'hybrid-keyboard-optimizer/data/wonderful_wizard_oz_cln.txt'


    time_start=time.time()
    best_layout, evolution_history = run_genetic_algorithm(
        text_file=TEXT_FILE,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        seed=SEED,
        use_known_distributions=False,
    )
    
    if best_layout:
        # Compare with QWERTY
        text_content = load_text_from_file(TEXT_FILE)
        bigrams = precompute_bigrams(text_content)
        
        qwerty = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"]
        ]
        
        dvorak = [
            "'", ',', '.', 'p', 'y', 'f', 'g', 'c', 'r', 'l',  
            'a', 'o', 'e', 'u', 'i', 'd', 'h', 't', 'n', 's',
            ';', 'q', 'j', 'k', 'x', 'b', 'm', 'w', 'v', 'z'   
        ]
        

        qwertz = [
            'q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p',  
            'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',  
            'y', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"   
        ]
        
        
        colemak = [
            'q', 'w', 'f', 'p', 'g', 'j', 'l', 'u', 'y', ';',  
            'a', 'r', 's', 't', 'd', 'h', 'n', 'e', 'i', 'o',  
            'z', 'x', 'c', 'v', 'b', 'k', 'm', ',', '.', "'"   
        ]
        
        qwerty_fitness = fitness_function_optimized([qwerty], bigrams)[0]
        qwertz_fitness = fitness_function_optimized([qwertz], bigrams)[0]
        dvorak_fitness = fitness_function_optimized([dvorak], bigrams)[0]
        colemak_fitness = fitness_function_optimized([colemak], bigrams)[0]

        best_fitness = evolution_history['best_fitness'][-1]
        
        improvement_qwerty = ((qwerty_fitness - best_fitness) / qwerty_fitness) * 100
        improvement_qwertz = ((qwertz_fitness - best_fitness) / qwertz_fitness) * 100
        improvement_dvorak = ((dvorak_fitness - best_fitness) / dvorak_fitness) * 100
        improvement_colemak = ((colemak_fitness - best_fitness) / colemak_fitness) * 100
        
        print(f"\nQWERTY Fitness: {qwerty_fitness:,.1f}")
        print_keyboard(qwerty, "QWERTY")
        print(f"\nQWERTZ Fitness: {qwertz_fitness:,.1f}")
        print_keyboard(qwertz, "QWERTZ")
        print(f"\nDVORAK Fitness: {dvorak_fitness:,.1f}")
        print_keyboard(dvorak, "Dvorak")
        print(f"\nCOLEMAK Fitness: {colemak_fitness:,.1f}")
        print_keyboard(colemak, "Colemak")
        
        print(f"\n\nBest Evolved Fitness: {best_fitness:,.1f}")
        print(f"Improvement: {improvement_qwerty:.1f}% better than QWERTY")
        print(f"Improvement: {improvement_qwertz:.1f}% better than QWERTZ")
        print(f"Improvement: {improvement_dvorak:.1f}% better than dvorak")
        print(f"Improvement: {improvement_colemak:.1f}% better than colemak")

        if CODIFICATION_FORMAT:
            print(best_layout)
        if LATEX_FORMAT:
            print_latex_data(evolution_history, GENERATIONS)
        print_keyboard(best_layout, "OPTIMIZED LAYOUT")
        time_end= time.time()
        print(f'Time consumed: {time_end-time_start}')
        
