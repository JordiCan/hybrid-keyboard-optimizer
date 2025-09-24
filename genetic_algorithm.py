import numpy as np
import random
from random import randint, shuffle

def init_population(pop_size, known_distributions=False):
    """
    Initialize a population of keyboard layouts.
    Each layout is represented as a list of characters.
    If known_distributions is True, include standard layouts like QWERTY, Dvorak, QWERTZ, and Colemak.
    Args:
        pop_size (int): The size of the population to generate.
        known_distributions (bool): Whether to include known keyboard layouts.
    Returns:
        list: A list of keyboard layouts, each represented as a list of characters.
    """


    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ',', '.', ';', "'"]
    
    keyboards = []
    
    if known_distributions:
        
        qwerty = [
            'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',  
            'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',  
            'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"   
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

        #Verifiy all the different known layouts contain 30 characters
        for i, layout in enumerate(keyboards):
            layout_names = ['QWERTY', 'Dvorak', 'QWERTZ', 'Colemak']
            if len(layout) != 30:
                print(f"Warning: {layout_names[i]} layout has {len(layout)} keys, expected 30")
            if set(layout)!=set(letters):
                missing=set(letters)-set(layout)
                extra=set(layout)-set(letters)
                if missing:
                    print(f"Warning: {layout_names[i]} layout is missing keys: {missing}")
                if extra:
                    print(f"Warning: {layout_names[i]} layout has extra keys: {extra}")

            

    for i in range(len(keyboards), pop_size):
        new_individual= random.sample(letters, len(letters))
        keyboards.append(new_individual)
    return keyboards



def get_finger_assigned(position):
    """
    Returns the finger assigned to a given key position.
    Args:
        position (int): The position of the key (0-29).
    Returns:
        tuple: (hand, finger, strength, row, col)
            - hand: 'L' (left) or 'R' (right)
            - finger: 0=pinky, 1=ring, 2=middle, 3=index
            - strength: 1-4 (4=strongest/index, 1=weakest/pinky)
            - row: 0-2 (top to bottom)
            - col: 0-9 (left to right)    
    """
    row=position // 10
    col=position % 10
    finger_map = {
        0: ('L', 0, 1), 1: ('L', 1, 2), 2: ('L', 2, 3), 3: ('L', 3, 4), 4: ('L', 3, 4),
        5: ('R', 3, 4), 6: ('R', 3, 4), 7: ('R', 2, 3), 8: ('R', 1, 2), 9: ('R', 0, 1)
    }

    hand, finger, strength = finger_map[col]
    return  hand, finger, strength, row, col 

def finger_penalty(pos1, pos2):
    """
    Computes the penalty based on finger strength, hand used and position, between two keys.
    Args:
        position (int): The position of the key (0-29).
    Returns:
        penalty (float): The computed penalty for the key position.
    """
    hand1, finger1, strength1, row1, col1 = get_finger_assigned(pos1)
    hand2, finger2, strength2, row2, col2 = get_finger_assigned(pos2)

    penalty=0
    if hand1 == hand2 and finger1 == finger2 and pos1 != pos2:
        penalty += 1.0
        if strength1<=2:
            penalty += 2.0
            
    elif hand1==hand2:
        penalty += 1.0

    penalty+= -1.0
    
    # === VERTICAL MOVEMENT PENALTIES ===
    row_diff = abs(row1 - row2)
    if row_diff == 1:  
        penalty += 0.2
        if strength1 <= 2 or strength2 <= 2:  
            penalty += 0.15
    elif row_diff == 2:  
        penalty += 0.8
        if strength1 <= 2 or strength2 <= 2:  
            penalty += 0.5
    
    # # === HORIZONTAL MOVEMENT PENALTIES ===
    # col_diff = abs(col1 - col2)
    # if col_diff >= 3:  
    #     penalty += 0.3
    #     if (hand1 != hand2 and 
    #         ((col1 in [0, 1] and col2 in [8, 9]) or 
    #          (col1 in [8, 9] and col2 in [0, 1]))):
    #         penalty += 0.4
    
    # === FINGER STRENGTH PENALTIES ===
    if finger1 == 0 or finger2 == 0:
        penalty += 0.15
    
    if finger1 == 1 or finger2 == 1:
        penalty += 0.1
    
    # === SAME COLUMN MOVEMENT ===
    if col1 == col2 and row_diff > 0:
        penalty += 0.3
        if col1 in [0, 9]:  
            penalty += 0.2
        elif col1 in [1, 8]:  
            penalty += 0.1
    
    return penalty

def euclidean_distance(pos1, pos2):
    """
    Computes the Euclidean distance between two key positions on the keyboard.
    Args:
        pos1 (int): The position of the first key (0-29).
        pos2 (int): The position of the second key (0-29).
    Returns:
        float: The Euclidean distance between the two keys.
    """
    _, _, _, row1, col1 = get_finger_assigned(pos1)
    _, _, _, row2, col2 = get_finger_assigned(pos2)
    return np.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)


def fitness_function(population, text):
    """
    Calculate fitness for each keyboard layout using optimized distance calculation
    Lower scores are better
    Args:
        population (list): A list of keyboard layouts, each represented as a list of characters.
        text (str): The text to evaluate the keyboard layouts against.
    Returns:
        list: A list of fitness scores corresponding to each keyboard layout.
    """
    results = []
    
    for keyboard in population:
        total_cost = 0
        
        for i in range(len(text) - 1):
            char1, char2 = text[i], text[i + 1]
            
            if char1 not in keyboard or char2 not in keyboard:
                continue
            
            pos1 = keyboard.index(char1)
            pos2 = keyboard.index(char2)
            
            # Skip if same position
            if pos1 == pos2:
                continue
            
            # Calculate optimized distance inline
            base_dist = euclidean_distance(pos1, pos2)
            finger_cost = finger_penalty(pos1, pos2)
            total_multiplier = max(1.0 + finger_cost, 0.1)
            
            total_cost += base_dist * total_multiplier
        
        results.append(total_cost)
    
    return results


def print_keyboard(layout, name="Keyboard"):
    """Print keyboard in 3x10 format"""
    print(f"\n{name}:")
    for row in range(3):
        row_keys = layout[row*10:(row+1)*10]
        print(" ".join(f"{key:>2}" for key in row_keys))
    
      
def test_system():
    """Test the optimized system"""
    population = init_population(4, known_distributions=True)
    sample_text = "the quick brown fox jumps over the lazy dog"
    
    fitness_scores = fitness_function(population, sample_text)
    layout_names = ['QWERTY', 'Dvorak', 'QWERTZ', 'Colemak']
    
    print("=== OPTIMIZED FITNESS RESULTS ===")
    for i, name in enumerate(layout_names):
        print(f"{name:8}: {fitness_scores[i]:8.2f}")
        print_keyboard(population[i], name)
        print("-" * 40)
    
    # Test random layouts too
    print("\n=== RANDOM LAYOUT COMPARISON ===")
    random_layouts = init_population(4, known_distributions=False)
    random_fitness = fitness_function(random_layouts, sample_text)
    
    for i, layout in enumerate(random_layouts):
        print("-" * 40)
        print(f"Random {i+1}: {random_fitness[i]:8.2f}")
        print_keyboard(layout, f"Random Layout {i+1}")
        print("-" * 40+ "\n")


if __name__ == "__main__":
    test_system()
    