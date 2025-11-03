import numpy as np
import random
import time
from collections import defaultdict
import os
from datetime import datetime

# Keyboard layouts globales
qwerty = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"]
]
dvorak = ["'", ',', '.', 'p', 'y', 'f', 'g', 'c', 'r', 'l',  
        'a', 'o', 'e', 'u', 'i', 'd', 'h', 't', 'n', 's',
        ';', 'q', 'j', 'k', 'x', 'b', 'm', 'w', 'v', 'z']
qwertz = ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p',  
        'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',  
        'y', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"]
colemak = ['q', 'w', 'f', 'p', 'g', 'j', 'l', 'u', 'y', ';',  
        'a', 'r', 's', 't', 'd', 'h', 'n', 'e', 'i', 'o',  
        'z', 'x', 'c', 'v', 'b', 'k', 'm', ',', '.', "'"]

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
    """Precompute all position-to-position costs (30x30)"""
    distances = np.zeros((30, 30))
    penalties = np.zeros((30, 30))
    
    for i in range(30):
        for j in range(30):
            distances[i, j] = euclidean_distance(i, j)
            penalties[i, j] = finger_penalty(i, j)
    
    return distances, penalties

# Global matrices
DISTANCE_MATRIX, PENALTY_MATRIX = precompute_cost_matrices()

# ============ TEXT PROCESSING ============
def load_text_from_file(filename, sample_size=None):
    """Load and optionally sample text"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read().lower()
        
        if sample_size and sample_size < len(text):
            text = text[:sample_size]
        
        return text
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def precompute_bigrams(text):
    """Precompute bigram frequencies"""
    bigrams = defaultdict(int)
    
    for i in range(len(text) - 1):
        bigram = (text[i], text[i + 1])
        bigrams[bigram] += 1
    
    return dict(bigrams)

def fitness_function_optimized(population, bigram_freq):
    """OPTIMIZED fitness using precomputed bigrams and cost matrices"""
    results = []
    
    for keyboard in population:
        pos_map = {char: idx for idx, char in enumerate(keyboard)}
        total_cost = 0
        
        for (char1, char2), freq in bigram_freq.items():
            if char1 not in pos_map or char2 not in pos_map:
                continue
            
            pos1 = pos_map[char1]
            pos2 = pos_map[char2]
            
            if pos1 == pos2:
                continue
            
            base_dist = DISTANCE_MATRIX[pos1, pos2]
            finger_cost = PENALTY_MATRIX[pos1, pos2]
            total_multiplier = max(1.0 + finger_cost, 0.1)
            
            total_cost += (base_dist * total_multiplier) * freq
        
        results.append(total_cost)
    
    return results

# ============ GENETIC OPERATORS ============
def init_population(pop_size, known_distributions=False):
    """Initialize population"""
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
            ',', '.', ';', "'"]
    
    keyboards = []
    
    if known_distributions:
        keyboards = [qwerty, dvorak, qwertz, colemak]
    
    for _ in range(len(keyboards), pop_size):
        keyboards.append(random.sample(letters, len(letters)))
    
    return keyboards

def crossover_two_point(parent1, parent2):
    """Two-point crossover for permutation encoding"""
    size = len(parent1)
    point1 = random.randint(1, size - 2)
    point2 = random.randint(point1 + 1, size - 1)
    
    child1 = [None] * size
    child1[point1:point2] = parent1[point1:point2]
    inherited1 = set(parent1[point1:point2])
    p2_filtered = [g for g in parent2 if g not in inherited1]
    child1[:point1] = p2_filtered[:point1]
    child1[point2:] = p2_filtered[point1:]
    
    child2 = [None] * size
    child2[point1:point2] = parent2[point1:point2]
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

# ============ UTILITY FUNCTIONS ============
def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"

def get_document_name(filepath):
    """Extract document name from filepath without extension"""
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]

# ============ MAIN GA ============
def run_genetic_algorithm(text_file, pop_size=50, generations=50, elite_rate=0.1, 
                        tournament_size=3, mutation_rate=0.15, 
                        use_known_distributions=True, seed=None, output_file=None):
    """Optimized genetic algorithm"""
    
    output_lines = []
    
    def log(text):
        print(text)
        if output_file:
            output_lines.append(text)
    
    log("=" * 60)
    log("OPTIMIZED GENETIC ALGORITHM FOR KEYBOARD LAYOUT")
    log("=" * 60)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    text = load_text_from_file(text_file)
    if text is None:
        return None, None, output_lines
    
    bigram_freq = precompute_bigrams(text)
    population = init_population(pop_size, use_known_distributions)
    
    elite_size = int(pop_size * elite_rate)
    
    log(f"Population: {pop_size} layouts")
    log(f"Elite size: {elite_size} ({elite_rate*100:.1f}%)")
    log(f"Tournament size: {tournament_size}")
    log(f"Mutation rate: {mutation_rate}")
    log("")
    
    history = {"best_fitness": [], "avg_fitness": []}
    best_layout_ever = None
    best_fitness_ever = float('inf')
    
    for gen in range(generations):
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
        
        log(f"Gen {gen+1:3}/{generations} | Best: {current_best:>12,.1f} | Avg: {current_avg:>12,.1f}")
        
        # Next generation (always crossover)
        next_generation = [layout.copy() for layout in sorted_population[:elite_size]]
        
        num_offspring_needed = pop_size - elite_size
        num_pairs = num_offspring_needed // 2
        offspring = []
        
        for _ in range(num_pairs):
            p1 = tournament_selection(population_with_fitness, tournament_size)
            p2 = tournament_selection(population_with_fitness, tournament_size)
            c1, c2 = crossover_two_point(p1, p2)
            offspring.extend([c1, c2])
        
        if len(offspring) < num_offspring_needed:
            p1 = tournament_selection(population_with_fitness, tournament_size)
            p2 = tournament_selection(population_with_fitness, tournament_size)
            c1, _ = crossover_two_point(p1, p2)
            offspring.append(c1)
        
        offspring = mutation(offspring[:num_offspring_needed], mutation_rate)
        next_generation.extend(offspring)
        population = next_generation
    
    log("\n" + "=" * 60)
    log("EVOLUTION COMPLETE")
    log("=" * 60)
    
    return best_layout_ever, history, output_lines

def print_keyboard(layout, name="Keyboard"):
    """Print keyboard layout"""
    lines = [f"\n{name}:"]
    for row in range(3):
        row_keys = layout[row*10:(row+1)*10]
        lines.append("  " + " ".join(f"{key:>2}" for key in row_keys))
    return lines

def print_latex_data(history, generations):
    """Print best fitness data in LaTeX format"""
    lines = ["\n" + "="*60]
    lines.append("DATOS PARA LATEX")
    lines.append("="*60)
    lines.append("\n% Generación y Best Fitness")
    lines.append("% coordinates {(gen, fitness)}")
    lines.append("")
    
    for gen in range(1, generations + 1):
        lines.append(f"({gen}, {history['best_fitness'][gen-1]:.1f})")
    
    lines.append("\n" + "="*60)
    return lines

def save_results(output_lines, filename):
    """Save results to 'hybrid-keyboard-optimizer/results/' directory"""
    results_dir = os.path.join(os.getcwd(), 'hybrid-keyboard-optimizer', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Results saved to: {filepath}")


def add_comparison_results(output_lines, best_layout, history, config, bigrams):
    """Add comparison results to output lines"""
    qwerty_fitness = fitness_function_optimized([qwerty], bigrams)[0]
    dvorak_fitness = fitness_function_optimized([dvorak], bigrams)[0]
    qwertz_fitness = fitness_function_optimized([qwertz], bigrams)[0]
    colemak_fitness = fitness_function_optimized([colemak], bigrams)[0]
    best_fitness = history['best_fitness'][-1]
    
    output_lines.append(f"\n\nQWERTY Fitness: {qwerty_fitness:,.1f}")
    output_lines.extend(print_keyboard(qwerty, "QWERTY"))
    output_lines.append(f"\nDVORAK Fitness: {dvorak_fitness:,.1f}")
    output_lines.extend(print_keyboard(dvorak, "DVORAK"))
    output_lines.append(f"\nQWERTZ Fitness: {qwertz_fitness:,.1f}")
    output_lines.extend(print_keyboard(qwertz, "QWERTZ"))
    output_lines.append(f"\nCOLEMAK Fitness: {colemak_fitness:,.1f}")
    output_lines.extend(print_keyboard(colemak, "COLEMAK"))
    
    output_lines.append(f"\n\nBest Evolved Fitness: {best_fitness:,.1f}")
    output_lines.append(f"Improvement vs QWERTY: {((qwerty_fitness - best_fitness) / qwerty_fitness) * 100:.2f}%")
    output_lines.append(f"Improvement vs DVORAK: {((dvorak_fitness - best_fitness) / dvorak_fitness) * 100:.2f}%")
    output_lines.append(f"Improvement vs QWERTZ: {((qwertz_fitness - best_fitness) / qwertz_fitness) * 100:.2f}%")
    output_lines.append(f"Improvement vs COLEMAK: {((colemak_fitness - best_fitness) / colemak_fitness) * 100:.2f}%")
    
    output_lines.append(f"\nLayout representation:")
    output_lines.append(str(best_layout))
    output_lines.extend(print_keyboard(best_layout, "OPTIMIZED LAYOUT"))
    output_lines.extend(print_latex_data(history, config['generations']))

# ============ EXPERIMENTS ============
def run_experiments():
    """Run systematic experiments varying one parameter at a time"""
    
    TEXT_FILE = 'hybrid-keyboard-optimizer/data/moby_dick_cln.txt'
    SEED = 123
    
    # Extract document name for filenames
    doc_name = get_document_name(TEXT_FILE)
    
    # BASE PARAMETERS
    BASE_CONFIG = {
        'pop_size': 100000,
        'generations': 150,
        'elite_rate': 0.15,  
        'tournament_size': 5,
        'mutation_rate': 0.15,
        'use_known_distributions': False
    }
    
    print("\n" + "="*70)
    print("STARTING EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Document: {doc_name}")
    print(f"Base configuration: {BASE_CONFIG}")
    print(f"Each experiment varies ONE parameter while keeping others constant")
    print("="*70 + "\n")
    
    # EXPERIMENT 1: Population Size
    print("\n" + "EXPERIMENT 1: POPULATION SIZE")
    print("-" * 70)
    pop_sizes = [1000, 10000, 100000, 1000000]
    
    for pop_size in pop_sizes:
        print(f"\nTesting population size: {pop_size:,}")
        config = BASE_CONFIG.copy()
        config['pop_size'] = pop_size
        
        time_start = time.time()
        best_layout, history, output_lines = run_genetic_algorithm(
            text_file=TEXT_FILE,
            seed=SEED,
            output_file=True,
            **config
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            text_content = load_text_from_file(TEXT_FILE)
            bigrams = precompute_bigrams(text_content)
            
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')

            filename = f"exp1_popsize_{pop_size}_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 2: Tournament Size (k)
    print("\n\n" + "EXPERIMENT 2: TOURNAMENT SIZE (k)")
    print("-" * 70)
    tournament_sizes = [2, 3, 5, 7, 10]
    
    for k in tournament_sizes:
        print(f"\nTesting tournament size k={k}")
        config = BASE_CONFIG.copy()
        config['tournament_size'] = k
        
        time_start = time.time()
        best_layout, history, output_lines = run_genetic_algorithm(
            text_file=TEXT_FILE,
            seed=SEED,
            output_file=True,
            **config
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            text_content = load_text_from_file(TEXT_FILE)
            bigrams = precompute_bigrams(text_content)
            
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')
            
            filename = f"exp2_tournament_k{k}_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 3: Mutation Rate
    print("\n\n" + "EXPERIMENT 3: MUTATION RATE")
    print("-" * 70)
    mutation_rates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75]
    
    for mut_rate in mutation_rates:
        print(f"\nTesting mutation rate: {mut_rate}")
        config = BASE_CONFIG.copy()
        config['mutation_rate'] = mut_rate
        
        time_start = time.time()
        best_layout, history, output_lines = run_genetic_algorithm(
            text_file=TEXT_FILE,
            seed=SEED,
            output_file=True,
            **config
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            text_content = load_text_from_file(TEXT_FILE)
            bigrams = precompute_bigrams(text_content)
            
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')

            filename = f"exp3_mutation_{mut_rate}_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 4: Elite Size (Generational Replacement %)
    print("\n\n" + "EXPERIMENT 4: ELITE SIZE (Generational Replacement %)")
    print("-" * 70)
    elite_percentages = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    
    for elite_pct in elite_percentages:
        elite_size = int(BASE_CONFIG['pop_size'] * elite_pct)
        print(f"\nTesting elite rate: {elite_pct*100:.0f}% ({elite_size:,} individuals)")
        config = BASE_CONFIG.copy()
        config['elite_rate'] = elite_pct
        
        time_start = time.time()
        best_layout, history, output_lines = run_genetic_algorithm(
            text_file=TEXT_FILE,
            seed=SEED,
            output_file=True,
            **config
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            text_content = load_text_from_file(TEXT_FILE)
            bigrams = precompute_bigrams(text_content)
            
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')

            filename = f"exp4_elite_{int(elite_pct*100)}pct_{doc_name}.txt"
            save_results(output_lines, filename)
    
    print("\n\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Results saved in 'results/' directory")
    print(f"Total experiments: {len(pop_sizes) + len(tournament_sizes) + len(mutation_rates) + len(elite_percentages)}")

# ============ MAIN ============
if __name__ == "__main__":
    print("\nKEYBOARD OPTIMIZER - EXPERIMENTAL SUITE\n")
    print("This script will run 4 series of experiments:")
    print("  1. Population Size: [1000, 10000, 100000, 1000000]")
    print("  2. Tournament Size (k): [2, 3, 5, 7, 10]")
    print("  3. Mutation Rate: [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75]")
    print("  4. Elite Rate: [5%, 10%, 15%, 20%, 30%, 50%]")
    print("\nEach experiment varies ONE parameter while keeping others constant.")
    print("All results saved to individual .txt files in 'results/' directory.\n")
        
    run_experiments()
