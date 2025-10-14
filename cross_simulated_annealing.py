import numpy as np
import random
import time
from collections import defaultdict
import os
from datetime import datetime

# ============ KEYBOARD LAYOUTS ============
qwerty = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',  
        'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',  
        'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"]
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
    - 'geometric': T_i+1 = k * T_i
    - 'logarithmic': T_i+1 = T_i / (1 + k*T_i)
    """
    if schedule_type == 'linear':
        return max(T_initial - iteration * k, 0.01)
    elif schedule_type == 'geometric':
        return T_initial * (k ** iteration)
    elif schedule_type == 'logarithmic':
        T_current = T_initial
        for _ in range(iteration):
            T_current = T_current / (1 + k * T_current)
        return T_current
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def simulated_annealing(text_file, bigram_freq, initial_layout='random', T_initial=1000, 
                       max_iterations=10000, schedule_type='geometric', k=0.95, 
                       seed=None, output_file=None):
    """
    Simulated Annealing for keyboard optimization
    """
    output_lines = []
    
    def log(text):
        print(text)
        if output_file:
            output_lines.append(text)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
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
    
    log("=" * 70)
    log("SIMULATED ANNEALING FOR KEYBOARD OPTIMIZATION")
    log("=" * 70)
    log(f"Initial layout: {initial_layout}")
    log(f"T_initial: {T_initial}")
    log(f"Max iterations: {max_iterations}")
    log(f"Schedule: {schedule_type}, k={k}")
    log(f"Initial fitness: {f_actual:,.1f}")
    log("=" * 70)
    
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
            acceptance_prob = np.exp(delta_f / T) if T > 0 else 0
            if random.random() < acceptance_prob:
                S_actual = S_nuevo
                f_actual = f_nuevo
                history['accepted_moves'] += 1
            else:
                history['rejected_moves'] += 1
        
        # Record history
        if i % 100 == 0 or i == max_iterations - 1:
            history['iteration'].append(i)
            history['temperature'].append(T)
            history['current_fitness'].append(f_actual)
            history['best_fitness'].append(f_mejor)
            
            if i % 5000 == 0:
                log(f"Iter {i:5}/{max_iterations} | T={T:8.2f} | "
                    f"Current: {f_actual:>12,.1f} | Best: {f_mejor:>12,.1f}")
    
    log("=" * 70)
    log("SIMULATED ANNEALING COMPLETE")
    log("=" * 70)
    log(f"Final best fitness: {f_mejor:,.1f}")
    log(f"Accepted moves: {history['accepted_moves']:,}")
    log(f"Rejected moves: {history['rejected_moves']:,}")
    log(f"Acceptance rate: {history['accepted_moves']/(history['accepted_moves']+history['rejected_moves'])*100:.1f}%")
    
    return S_mejor, history, output_lines

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

def print_keyboard(layout, name="Keyboard"):
    """Print keyboard layout"""
    lines = [f"\n{name}:"]
    for row in range(3):
        row_keys = layout[row*10:(row+1)*10]
        lines.append("  " + " ".join(f"{key:>2}" for key in row_keys))
    return lines

def print_latex_data(history):
    """Print best fitness data in LaTeX format"""
    lines = ["\n" + "="*60]
    lines.append("DATOS PARA LATEX")
    lines.append("="*60)
    lines.append("\n% Iteration and Best Fitness")
    lines.append("% coordinates {(iteration, fitness)}")
    lines.append("")
    
    for i, iter_num in enumerate(history['iteration']):
        lines.append(f"({iter_num}, {history['best_fitness'][i]:.1f})")
    
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
    qwerty_fitness = fitness_function(qwerty, bigrams)
    dvorak_fitness = fitness_function(dvorak, bigrams)
    qwertz_fitness = fitness_function(qwertz, bigrams)
    colemak_fitness = fitness_function(colemak, bigrams)
    best_fitness = history['best_fitness'][-1]
    
    output_lines.append(f"\n\nQWERTY Fitness: {qwerty_fitness:,.1f}")
    output_lines.extend(print_keyboard(qwerty, "QWERTY"))
    output_lines.append(f"\nDVORAK Fitness: {dvorak_fitness:,.1f}")
    output_lines.extend(print_keyboard(dvorak, "DVORAK"))
    output_lines.append(f"\nQWERTZ Fitness: {qwertz_fitness:,.1f}")
    output_lines.extend(print_keyboard(qwertz, "QWERTZ"))
    output_lines.append(f"\nCOLEMAK Fitness: {colemak_fitness:,.1f}")
    output_lines.extend(print_keyboard(colemak, "COLEMAK"))
    
    output_lines.append(f"\n\nBest SA Fitness: {best_fitness:,.1f}")
    output_lines.append(f"Improvement vs QWERTY: {((qwerty_fitness - best_fitness) / qwerty_fitness) * 100:.2f}%")
    output_lines.append(f"Improvement vs DVORAK: {((dvorak_fitness - best_fitness) / dvorak_fitness) * 100:.2f}%")
    output_lines.append(f"Improvement vs QWERTZ: {((qwertz_fitness - best_fitness) / qwertz_fitness) * 100:.2f}%")
    output_lines.append(f"Improvement vs COLEMAK: {((colemak_fitness - best_fitness) / colemak_fitness) * 100:.2f}%")
    
    output_lines.append(f"\nLayout representation:")
    output_lines.append(str(best_layout))
    output_lines.extend(print_keyboard(best_layout, "OPTIMIZED SA LAYOUT"))
    output_lines.extend(print_latex_data(history))

# ============ EXPERIMENTS ============
def run_experiments():
    """Run systematic experiments varying temperature parameters"""
    
    TEXT_FILE = 'hybrid-keyboard-optimizer/data/moby_dick_cln.txt'
    SEED = 123
    
    # Extract document name for filenames
    doc_name = get_document_name(TEXT_FILE)
    
    # Load text and precompute bigrams once
    print("\nLoading text and computing bigrams...")
    text_content = load_text_from_file(TEXT_FILE)
    if text_content is None:
        return
    bigrams = precompute_bigrams(text_content)
    print(f"✓ Text loaded: {len(text_content):,} characters")
    print(f"✓ Bigrams computed: {len(bigrams):,} unique pairs\n")
    
    # BASE PARAMETERS
    BASE_CONFIG = {
        'max_iterations': 50000,
        'initial_layout': 'random',
        'schedule_type': 'geometric',
        'k': 0.95
    }
    
    print("\n" + "="*70)
    print("STARTING SIMULATED ANNEALING EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Document: {doc_name}")
    print(f"Base configuration: {BASE_CONFIG}")
    print(f"Each experiment varies temperature parameters")
    print("="*70 + "\n")
    
    # EXPERIMENT 1: Initial Temperature (T_initial) with Geometric Schedule
    print("\n" + "EXPERIMENT 1: INITIAL TEMPERATURE (T_initial) - GEOMETRIC")
    print("-" * 70)
    T_initials_geom = [100, 500, 1000, 5000, 10000]
    
    for T_init in T_initials_geom:
        print(f"\nTesting T_initial: {T_init}")
        config = BASE_CONFIG.copy()
        config['T_initial'] = T_init
        
        time_start = time.time()
        best_layout, history, output_lines = simulated_annealing(
            text_file=TEXT_FILE,
            bigram_freq=bigrams,
            T_initial=T_init,
            seed=SEED,
            output_file=True,
            **{k: v for k, v in config.items() if k != 'T_initial'}
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')
            
            filename = f"sa_exp1_Tinit_{T_init}_geometric_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 2: Initial Temperature (T_initial) with Linear Schedule
    print("\n\n" + "EXPERIMENT 2: INITIAL TEMPERATURE (T_initial) - LINEAR")
    print("-" * 70)
    T_initials_linear = [1000, 5000, 10000, 50000]
    
    for T_init in T_initials_linear:
        print(f"\nTesting T_initial: {T_init} (linear)")
        config = BASE_CONFIG.copy()
        config['T_initial'] = T_init
        config['schedule_type'] = 'linear'
        config['k'] = 0.5  # Linear decay rate
        
        time_start = time.time()
        best_layout, history, output_lines = simulated_annealing(
            text_file=TEXT_FILE,
            bigram_freq=bigrams,
            T_initial=T_init,
            schedule_type='linear',
            k=0.5,
            seed=SEED,
            output_file=True,
            max_iterations=config['max_iterations'],
            initial_layout=config['initial_layout']
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')
            
            filename = f"sa_exp2_Tinit_{T_init}_linear_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 3: Cooling Rate (k) with Geometric Schedule
    print("\n\n" + "EXPERIMENT 3: COOLING RATE (k) - GEOMETRIC")
    print("-" * 70)
    cooling_rates = [0.80, 0.85, 0.90, 0.95, 0.99]
    
    for k_val in cooling_rates:
        print(f"\nTesting cooling rate k={k_val}")
        config = BASE_CONFIG.copy()
        config['T_initial'] = 1000
        config['k'] = k_val
        
        time_start = time.time()
        best_layout, history, output_lines = simulated_annealing(
            text_file=TEXT_FILE,
            bigram_freq=bigrams,
            T_initial=1000,
            k=k_val,
            seed=SEED,
            output_file=True,
            **{k: v for k, v in config.items() if k not in ['T_initial', 'k']}
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')
            
            filename = f"sa_exp3_k_{k_val}_geometric_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 4: Logarithmic Schedule with different k values
    print("\n\n" + "EXPERIMENT 4: LOGARITHMIC SCHEDULE - VARYING k")
    print("-" * 70)
    k_logarithmic = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    for k_val in k_logarithmic:
        print(f"\nTesting logarithmic schedule with k={k_val}")
        config = BASE_CONFIG.copy()
        config['T_initial'] = 1000
        config['schedule_type'] = 'logarithmic'
        config['k'] = k_val
        
        time_start = time.time()
        best_layout, history, output_lines = simulated_annealing(
            text_file=TEXT_FILE,
            bigram_freq=bigrams,
            T_initial=1000,
            schedule_type='logarithmic',
            k=k_val,
            seed=SEED,
            output_file=True,
            max_iterations=config['max_iterations'],
            initial_layout=config['initial_layout']
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')
            
            filename = f"sa_exp4_k_{k_val}_logarithmic_{doc_name}.txt"
            save_results(output_lines, filename)
    
    # EXPERIMENT 5: Different Initial Layouts
    print("\n\n" + "EXPERIMENT 5: INITIAL LAYOUT")
    print("-" * 70)
    initial_layouts = ['random', 'qwerty', 'dvorak', 'qwertz', 'colemak']
    
    for layout in initial_layouts:
        print(f"\nTesting initial layout: {layout}")
        config = BASE_CONFIG.copy()
        config['T_initial'] = 1000
        config['initial_layout'] = layout
        
        time_start = time.time()
        best_layout, history, output_lines = simulated_annealing(
            text_file=TEXT_FILE,
            bigram_freq=bigrams,
            initial_layout=layout,
            T_initial=1000,
            seed=SEED,
            output_file=True,
            **{k: v for k, v in config.items() if k not in ['T_initial', 'initial_layout']}
        )
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        if best_layout:
            add_comparison_results(output_lines, best_layout, history, config, bigrams)
            output_lines.append(f'\nTime consumed: {elapsed_time:.2f}s ({seconds_to_hms(elapsed_time)})')
            
            filename = f"sa_exp5_initial_{layout}_{doc_name}.txt"
            save_results(output_lines, filename)
    
    print("\n\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Results saved in 'hybrid-keyboard-optimizer/results/' directory")
    total_experiments = (len(T_initials_geom) + len(T_initials_linear) + 
                        len(cooling_rates) + len(k_logarithmic) + len(initial_layouts))
    print(f"Total experiments: {total_experiments}")

# ============ MAIN ============
if __name__ == "__main__":
    print("\nKEYBOARD OPTIMIZER - SIMULATED ANNEALING EXPERIMENTAL SUITE\n")
    print("This script will run 5 series of experiments:")
    print("  1. Initial Temperature (Geometric): [100, 500, 1000, 5000, 10000]")
    print("  2. Initial Temperature (Linear): [1000, 5000, 10000, 50000]")
    print("  3. Cooling Rate k (Geometric): [0.80, 0.85, 0.90, 0.95, 0.99]")
    print("  4. Logarithmic Schedule k: [0.001, 0.005, 0.01, 0.05, 0.1]")
    print("  5. Initial Layout: ['random', 'qwerty', 'dvorak', 'qwertz', 'colemak']")
    print("\nEach experiment varies ONE parameter while keeping others constant.")
    print("All results saved to individual .txt files in 'results/' directory.\n")
        
    run_experiments()