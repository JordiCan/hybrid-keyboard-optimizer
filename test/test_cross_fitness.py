# keyboard_fitness.py
import numpy as np
from collections import defaultdict

# ============ FINGER & DISTANCE FUNCTIONS ============
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

DISTANCE_MATRIX, PENALTY_MATRIX = precompute_cost_matrices()

# ============ TEXT & BIGRAM FUNCTIONS ============
def load_text(filename):
    """Load text from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().lower()

def precompute_bigrams(text):
    """Compute bigram frequencies"""
    bigrams = defaultdict(int)
    for i in range(len(text) - 1):
        bigrams[(text[i], text[i+1])] += 1
    return dict(bigrams)

# ============ FITNESS FUNCTION ============
def fitness_function(keyboard_layout, bigram_freq):
    """Compute fitness for a single keyboard layout"""
    pos_map = {char: idx for idx, char in enumerate(keyboard_layout)}
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

if __name__ == "__main__":
    moby_dick=["'", ';', ',', 'd', 'y', 'l', 'b', 'w', 'x', 'z', '.', 'g', 'i', 'e', 'a', 'n', 't', 'h', 'm', 'v', 'j', 'k', 'p', 'u', 'o', 'r', 's', 'c', 'f', 'q']
    wizard_oz=['j', 'p', 'c', 't', 'u', 'h', 's', 'w', 'x', 'q', '.', 'y', 'e', 'a', 'i', 'r', 'o', 'l', 'm', 'b', ';', 'v', ',', 'g', 'f', 'n', 'd', 'k', 'z', "'"]
    
    moby_text= "data/moby_dick_cln.txt"
    wizard_text= "data/wonderful_wizard_oz_cln.txt"
    moby_text= load_text(moby_text)
    wizard_text= load_text(wizard_text)
    bigrams_moby = precompute_bigrams(moby_text)
    bigrams_wizard= precompute_bigrams(wizard_text)

    # Compute fitness on Moby Dick text
    fitness_moby_on_moby = fitness_function(moby_dick, bigrams_moby)
    fitness_wizard_on_moby = fitness_function(wizard_oz, bigrams_moby)

    # Compute fitness on Wizard of Oz text
    fitness_moby_on_wizard = fitness_function(moby_dick, bigrams_wizard)
    fitness_wizard_on_wizard = fitness_function(wizard_oz, bigrams_wizard)

    # Print results nicely
    print("\n=== Fitness on Moby Dick Text ===")
    print(f"Moby Dick layout:       {fitness_moby_on_moby:,.1f}")
    print(f"Wizard of Oz layout:    {fitness_wizard_on_moby:,.1f}")

    print("\n=== Fitness on Wizard of Oz Text ===")
    print(f"Moby Dick layout:       {fitness_moby_on_wizard:,.1f}")
    print(f"Wizard of Oz layout:    {fitness_wizard_on_wizard:,.1f}")


  
