import numpy as np
import random
import time
from collections import defaultdict
import os

# ============ KEYBOARD LAYOUTS ============
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


# Best from Exp1 - Moby Dick (Pop Size: 100000, Fitness: 769,434.6)
exp1_best_moby_dick = ['x', 'v', 'w', 'c', 'h', 'i', 't', 'k', 'z', "'", 'f', 'm', 'r', 's', 'n', 'e', 'a', 'o', ',', '.', 'q', 'b', 'p', 'l', 'd', 'g', 'u', 'y', ';', 'j']

# Best from Exp1 - Wizard of Oz (Pop Size: 1000000, Fitness: 121,733.0)
exp1_best_wizard_oz = ['z', 'k', 'w', 's', 'h', 'i', 't', 'c', ',', ';', 'v', 'd', 'n', 'l', 'r', 'e', 'a', 'o', 'y', '.', 'q', 'x', 'b', 'f', 'm', 'g', 'u', 'p', 'j', "'"]

# Best from Exp2 - Moby Dick (Tournament k: 5, Fitness: 769,434.6)
exp2_best_moby_dick = ['x', 'v', 'w', 'c', 'h', 'i', 't', 'k', 'z', "'", 'f', 'm', 'r', 's', 'n', 'e', 'a', 'o', ',', '.', 'q', 'b', 'p', 'l', 'd', 'g', 'u', 'y', ';', 'j']

# Best from Exp2 - Wizard of Oz (Tournament k: 10, Fitness: 122,030.3)
exp2_best_wizard_oz = ['x', 'v', 'c', 'w', 'h', 'i', 't', 'k', 'z', "'", 'f', 'm', 'r', 's', 'n', 'e', 'a', 'o', 'g', '.', 'q', 'b', 'p', 'l', 'd', ',', 'y', 'u', 'j', ';']


# Best from Exp3 - Moby Dick (Mutation: 0.15, Fitness: 769,434.6)
exp3_best_moby_dick = ['x', 'v', 'w', 'c', 'h', 'i', 't', 'k', 'z', "'", 'f', 'm', 'r', 's', 'n', 'e', 'a', 'o', ',', '.', 'q', 'b', 'p', 'l', 'd', 'g', 'u', 'y', ';', 'j']

# Best from Exp3 - Wizard of Oz (Mutation: 0.5, Fitness: 123,341.2)
exp3_best_wizard_oz = ['q', 'p', 'u', 't', 'i', 'h', 's', 'c', 'b', 'x', '.', 'y', 'o', 'a', 'e', 'n', 'r', 'l', 'm', 'v', ';', 'j', ',', 'g', 'k', 'w', 'd', 'f', 'z', "'"]


# Best from Exp4 - Moby Dick (Elite: 20pct, Fitness: 765,886.5)
exp4_best_moby_dick = ["'", 'z', 'w', 't', 'i', 'h', 'c', 'f', 'k', 'x', '.', ',', 'o', 'a', 'e', 'n', 's', 'r', 'm', 'v', 'j', ';', 'y', 'u', 'g', 'd', 'l', 'p', 'b', 'q']

# Best from Exp4 - Wizard of Oz (Elite: 5pct, Fitness: 120,633.7)
exp4_best_wizard_oz = ['j', ';', 'p', 'y', ',', 'd', 'l', 'm', 'b', "'", '.', 'u', 'o', 'a', 'e', 'n', 's', 'r', 'f', 'v', 'q', 'z', 'g', 't', 'i', 'h', 'w', 'c', 'k', 'x']


# BEST OVERALL - Moby Dick (exp4_elite_20pct, Fitness: 765,886.5)
ga_best_moby_dick = ["'", 'z', 'w', 't', 'i', 'h', 'c', 'f', 'k', 'x', '.', ',', 'o', 'a', 'e', 'n', 's', 'r', 'm', 'v', 'j', ';', 'y', 'u', 'g', 'd', 'l', 'p', 'b', 'q']

# BEST OVERALL - Wizard of Oz (exp4_elite_5pct, Fitness: 120,633.7)
ga_best_wizard_oz = ['j', ';', 'p', 'y', ',', 'd', 'l', 'm', 'b', "'", '.', 'u', 'o', 'a', 'e', 'n', 's', 'r', 'f', 'v', 'q', 'z', 'g', 't', 'i', 'h', 'w', 'c', 'k', 'x']

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

def fitness_function(keyboard, bigram_freq):
    """Calculate fitness for a single keyboard layout"""
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
    
    return total_cost

# ============ SIMULATED ANNEALING ============
def get_random_neighbor(layout):
    """Generate neighbor by swapping two random positions"""
    neighbor = layout.copy()
    pos1, pos2 = np.random.choice(30, 2, replace=False)
    neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
    return neighbor

def temperature_schedule(T_initial, iteration, schedule_type='geometric', k=0.95):
    """
    Calculate temperature based on schedule type
    
    schedule_type options:
    - 'linear': T_i+1 = T_initial - i*k
    - 'geometric': T_i+1 = k * T_i (default, k typically 0.8-0.99)
    - 'logarithmic': T_i+1 = T_i / (1 + k*T_i)
    """
    if schedule_type == 'linear':
        return max(T_initial - iteration * k, 0.01)
    elif schedule_type == 'geometric':
        return T_initial * (k ** iteration)
    elif schedule_type == 'logarithmic':
        # Enfriamiento logarítmico más eficiente (sin bucle)
        # Aproximación: T_i ≈ T_initial / (1 + k * T_initial * iteration)
        return T_initial / (1 + k * T_initial * iteration)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def simulated_annealing(text_file, initial_layout='random', T_initial=1000, 
                       max_iterations=100000, schedule_type='geometric', k=0.95, 
                       seed=123, verbose=True):
    """
    Simulated Annealing for keyboard optimization
    
    Parameters:
    - text_file: path to text file for bigram analysis
    - initial_layout: 'random', 'qwerty', 'dvorak', 'qwertz', 'colemak'
    - T_initial: initial temperature
    - max_iterations: maximum number of iterations
    - schedule_type: 'linear', 'geometric', 'logarithmic'
    - k: cooling parameter
    - seed: random seed for reproducibility
    - verbose: print progress
    """
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load text and compute bigrams
    text = load_text_from_file(text_file)
    if text is None:
        return None, None
    
    bigram_freq = precompute_bigrams(text)
    
    # Initialize solution
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
            ',', '.', ';', "'"]
    
    if initial_layout == 'random':
        S_actual = random.sample(letters, len(letters))
    elif initial_layout == 'qwerty':
        S_actual = qwerty.copy()
    elif initial_layout == 'dvorak':
        S_actual = dvorak.copy()
    elif initial_layout == 'qwertz':
        S_actual = qwertz.copy()
    elif initial_layout == 'colemak':
        S_actual = colemak.copy()
    else:
        S_actual = random.sample(letters, len(letters))
    
    S_mejor = S_actual.copy()
    f_actual = fitness_function(S_actual, bigram_freq)
    f_mejor = f_actual
    
    history = {
        'iteration': [],
        'temperature': [],
        'current_fitness': [],
        'best_fitness': [],
        'accepted_moves': 0,
        'rejected_moves': 0
    }
    
    if verbose:
        print("=" * 70)
        print("SIMULATED ANNEALING FOR KEYBOARD OPTIMIZATION")
        print("=" * 70)
        print(f"Initial layout: {initial_layout}")
        print(f"T_initial: {T_initial}")
        print(f"Max iterations: {max_iterations}")
        print(f"Schedule: {schedule_type}, k={k}")
        print(f"Initial fitness: {f_actual:,.1f}")
        print("=" * 70)
    
    # Main SA loop
    for i in range(max_iterations):
        T = temperature_schedule(T_initial, i, schedule_type, k)
        
        # Generate random neighbor
        S_nuevo = get_random_neighbor(S_actual)
        f_nuevo = fitness_function(S_nuevo, bigram_freq)
        
        # Calculate delta (minimization: actual - nuevo)
        delta_f = f_actual - f_nuevo
        
        # Accept or reject move
        if delta_f > 0:  # Improvement
            S_actual = S_nuevo
            f_actual = f_nuevo
            history['accepted_moves'] += 1
            
            if f_nuevo < f_mejor:
                S_mejor = S_nuevo.copy()
                f_mejor = f_nuevo
        else:  # Worsening
            # Accept with probability e^(delta_f/T)
            # delta_f es negativo, así que e^(delta_f/T) < 1
            if T > 0:
                acceptance_prob = np.exp(delta_f / T)
                if random.random() < acceptance_prob:
                    S_actual = S_nuevo
                    f_actual = f_nuevo
                    history['accepted_moves'] += 1
                else:
                    history['rejected_moves'] += 1
            else:
                history['rejected_moves'] += 1
        
        # Record history every 100 iterations
        if i % 100 == 0 or i == max_iterations - 1:
            history['iteration'].append(i)
            history['temperature'].append(T)
            history['current_fitness'].append(f_actual)
            history['best_fitness'].append(f_mejor)
            
            if verbose and i % 1000 == 0:
                acc_rate = history['accepted_moves']/(history['accepted_moves']+history['rejected_moves'])*100 if (history['accepted_moves']+history['rejected_moves']) > 0 else 0
                print(f"Iter {i:5}/{max_iterations} | T={T:8.2f} | "
                    f"Current: {f_actual:>12,.1f} | Best: {f_mejor:>12,.1f} | Acc: {acc_rate:5.1f}%")
    
    if verbose:
        print("=" * 70)
        print("SIMULATED ANNEALING COMPLETE")
        print("=" * 70)
        print(f"Final best fitness: {f_mejor:,.1f}")
        print(f"Accepted moves: {history['accepted_moves']:,}")
        print(f"Rejected moves: {history['rejected_moves']:,}")
        print(f"Acceptance rate: {history['accepted_moves']/(history['accepted_moves']+history['rejected_moves'])*100:.1f}%")
    
    return S_mejor, history

# ============ UTILITY FUNCTIONS ============
def print_keyboard(layout, name="Keyboard"):
    """Print keyboard layout"""
    print(f"\n{name}:")
    for row in range(3):
        row_keys = layout[row*10:(row+1)*10]
        print("  " + " ".join(f"{key:>2}" for key in row_keys))

def compare_layouts(best_layout, history, text_file):
    """Compare best layout with standard layouts and all experimental GA layouts"""
    text = load_text_from_file(text_file)
    bigrams = precompute_bigrams(text)
    
    # Determine which dataset we're using
    is_moby_dick = 'moby' in text_file.lower()
    
    # Standard layouts
    layouts = {
        'QWERTY': qwerty,
        'DVORAK': dvorak,
        'QWERTZ': qwertz,
        'COLEMAK': colemak,
    }
    
    # Add experimental GA layouts based on dataset
    if is_moby_dick:
        layouts.update({
            'GA Exp1 (Pop=100k)': exp1_best_moby_dick,
            'GA Exp2 (Tour k=5)': exp2_best_moby_dick,
            'GA Exp3 (Mut=0.15)': exp3_best_moby_dick,
            'GA Exp4 (Elite=20%)': exp4_best_moby_dick,
            'GA BEST OVERALL': ga_best_moby_dick,
        })
    else:  # Wizard of Oz
        layouts.update({
            'GA Exp1 (Pop=1M)': exp1_best_wizard_oz,
            'GA Exp2 (Tour k=10)': exp2_best_wizard_oz,
            'GA Exp3 (Mut=0.5)': exp3_best_wizard_oz,
            'GA Exp4 (Elite=5%)': exp4_best_wizard_oz,
            'GA BEST OVERALL': ga_best_wizard_oz,
        })
    
    # Calculate fitness for all layouts
    results = []
    print("\n" + "=" * 80)
    print("LAYOUT COMPARISON")
    print("=" * 80)
    
    for name, layout in layouts.items():
        fitness = fitness_function(layout, bigrams)
        results.append((name, layout, fitness))
        print(f"\n{name} Fitness: {fitness:,.1f}")
        print_keyboard(layout, name)
    
    # Add SA result
    sa_fitness = history['best_fitness'][-1]
    results.append(('SA OPTIMIZED', best_layout, sa_fitness))
    print(f"\n\nSA OPTIMIZED Fitness: {sa_fitness:,.1f}")
    print_keyboard(best_layout, "SA OPTIMIZED LAYOUT")
    
    # Sort by fitness (lower is better)
    results.sort(key=lambda x: x[2])
    
    # Print ranking table
    print("\n" + "=" * 80)
    print("RANKING (Lower fitness is better)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Layout':<25} {'Fitness':>15} {'vs Best':>12}")
    print("-" * 80)
    
    best_fitness = results[0][2]
    for rank, (name, layout, fitness) in enumerate(results, 1):
        improvement = ((fitness - best_fitness) / best_fitness) * 100
        print(f"{rank:<6} {name:<25} {fitness:>15,.1f} {improvement:>11.2f}%")
    
    # Print improvements of SA vs reference layouts
    print("\n" + "=" * 80)
    print("SA IMPROVEMENTS vs REFERENCE LAYOUTS")
    print("=" * 80)
    
    reference_layouts = ['QWERTY', 'DVORAK', 'QWERTZ', 'COLEMAK', 'GA BEST OVERALL']
    for name, _, fitness in results:
        if name in reference_layouts:
            improvement = ((fitness - sa_fitness) / fitness) * 100
            status = "✓ Better" if improvement > 0 else "✗ Worse"
            print(f"vs {name:<20}: {improvement:>7.2f}% {status}")

# ============ MAIN ============
if __name__ == "__main__":
    TEXT_FILE = 'hybrid-keyboard-optimizer/data/moby_dick_cln.txt'
    
    time_start = time.time()
    
    # PARÁMETROS AJUSTADOS SEGÚN EL PDF DEL PROFESOR:
    # El PDF muestra que la temperatura debe bajar MUY lentamente
    # Con geometric k=0.9999: T baja de 1000 a ~6 en 50k iteraciones
    # Con logarithmic k muy pequeño: enfriamiento ultra-lento
    
    best_layout, history = simulated_annealing(
        text_file=TEXT_FILE,
        initial_layout='random',  # 'random', 'qwerty', 'dvorak', 'qwertz', 'colemak'
        T_initial=10000,  # T_initial MÁS ALTO para más exploración
        max_iterations=50000,
        schedule_type='geometric',  # 'linear', 'geometric', 'logarithmic'
        k=0.99992,  # Para geometric: 0.9999-0.99999 (MUY cercano a 1 = MUY lento)
                    # Para logarithmic: 0.000001-0.00001 (MUY pequeño = MUY lento)
                    # Para linear: k pequeño (0.1-1.0)
        seed=123,
        verbose=True
    )
    
    time_end = time.time()
    
    if best_layout:
        compare_layouts(best_layout, history, TEXT_FILE)
        print(f"\n\nTime consumed: {time_end - time_start:.2f}s")