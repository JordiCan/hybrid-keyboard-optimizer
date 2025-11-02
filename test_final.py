import numpy as np
from collections import defaultdict

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
    _, _, _, row1, col1 = get_finger_assigned(pos1)
    _, _, _, row2, col2 = get_finger_assigned(pos2)
    return np.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)

def finger_penalty(pos1, pos2):
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
    distances = np.zeros((30, 30))
    penalties = np.zeros((30, 30))
    
    for i in range(30):
        for j in range(30):
            distances[i, j] = euclidean_distance(i, j)
            penalties[i, j] = finger_penalty(i, j)
    
    return distances, penalties

DISTANCE_MATRIX, PENALTY_MATRIX = precompute_cost_matrices()

def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().lower()

def precompute_bigrams(text):
    bigrams = defaultdict(int)
    for i in range(len(text) - 1):
        bigrams[(text[i], text[i+1])] += 1
    return dict(bigrams)

def fitness_function(keyboard_layout, bigram_freq):
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

def evaluate_layouts(layouts, bigram_freq):
    results = {}
    for name, layout in layouts.items():
        results[name] = fitness_function(layout, bigram_freq)
    
    best_layout = min(results, key=results.get)
    return results, best_layout

if __name__ == "__main__":
    layouts = {
        "Moby dick" : [
            "z", "w", "c", "d", "n", "i", "h", "g", "k", ";",
            "v", "m", "r", "s", "t", "o", "a", "e", ",", ".",
            "j", "q", "b", "f", "l", "u", "p", "y", "x", "'"
        ],
        "Wizard of Oz": [
        "'", "b", "m", "l", "d", ",", "y", "p", ";", "j",
        "v", "f", "r", "s", "n", "e", "a", "o", "u", ".",
        "x", "k", "c", "w", "h", "i", "t", "g", "z", "q"
    ]
    }


    text_file = "data/guia_presentacion_cln.txt"
    text = load_text(text_file)
    bigrams = precompute_bigrams(text)

    results, best = evaluate_layouts(layouts, bigrams)

    print("=== Layout Fitness Scores ===")
    for name, score in results.items():
        print(f"{name}: {score:,.1f}")

    print(f"\nBest layout: {best} with fitness {results[best]:,.1f}")
