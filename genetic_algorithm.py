import numpy as np
import random
from random import randint, shuffle
import matplotlib.pyplot as plt


def load_text_from_file(filename):
    """Load text from file, handle encoding issues"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().lower()
    except FileNotFoundError:
        print(f"File {filename} not found. Using default text.")
        return None

    except Exception as e:
        print(f"Error reading file: {e}. Using default text.")
        return None

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


def selection(population, text, selection_method="elitist"):
    """
    Select and rank population based on fitness
    
    Args:
        population: List of keyboard layouts
        text: Text to evaluate fitness on
        selection_method: "elitist" (rank by fitness) or "tournament" 
    
    Returns:
        tuple: (selected_population, fitness_scores) - both sorted by fitness (best first)
    """
    fitness_scores = fitness_function(population, text)
    
    if selection_method == "elitist":
        sorted_indices = np.argsort(fitness_scores)
        selected_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        return selected_population, sorted_fitness
    
    elif selection_method == "tournament":
        selected_population = []
        selected_fitness = []
        
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected_population.append(population[winner_idx])
            selected_fitness.append(fitness_scores[winner_idx])
        
        return selected_population, selected_fitness
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


# def crossover(parent1, parent2):
#     """

#     Generate the offspring from two parents, based on the cross of two different segments.    
#     Args:
#         parent1, parent2: Parent keyboard layouts (permutations)
    
#     Returns:
#         list: Child keyboard layout (valid permutation)
#     """
#     size = len(parent1)
#     child = [None] * size
    
#     start = np.random.randint(0, size)
#     end = np.random.randint(start + 1, size + 1)
    
#     child[start:end] = parent1[start:end]
    
#     mapping = {}
#     for i in range(start, end):
#         mapping[parent1[i]] = parent2[i]
    
#     for i in range(size):
#         if child[i] is None:
#             candidate = parent2[i]
#             while candidate in child[start:end]:
#                 candidate = mapping[candidate]
#             child[i] = candidate
    
#     return child


# REEMPLAZA tu función crossover actual con esta:
def crossover(parent1, parent2):
    """
    Crossover que genera DOS hijos complementarios
    
    Args:
        parent1, parent2: Parent keyboard layouts (permutations)
    
    Returns:
        tuple: (child1, child2) - Dos hijos válidos
    """
    size = len(parent1)
    
    # Seleccionar punto de división (más conservador para mejor herencia)
    split_point = random.randint(size // 3, 2 * size // 3)  # División entre 33% y 67%
    
    # === CREAR HIJO 1: Primera mitad P1 + Segunda mitad P2 ===
    child1 = [None] * size
    child1[:split_point] = parent1[:split_point]
    inherited_from_p1 = set(parent1[:split_point])
    
    # Llenar resto con parent2 en orden
    p2_filtered = [gene for gene in parent2 if gene not in inherited_from_p1]
    child1[split_point:split_point+len(p2_filtered)] = p2_filtered
    
    # === CREAR HIJO 2: Primera mitad P2 + Segunda mitad P1 ===
    child2 = [None] * size
    child2[:split_point] = parent2[:split_point]
    inherited_from_p2 = set(parent2[:split_point])
    
    # Llenar resto con parent1 en orden
    p1_filtered = [gene for gene in parent1 if gene not in inherited_from_p2]
    child2[split_point:split_point+len(p1_filtered)] = p1_filtered
    
    return child1, child2


def fix_duplicates(child, parent1, parent2):
    """
    Fix duplicate characters in child by replacing with missing ones
    """
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ',', '.', ';', "'"]
    
    seen = set()
    duplicates = []
    
    for i, char in enumerate(child):
        if char in seen:
            duplicates.append(i)
        else:
            seen.add(char)
    
    missing = [char for char in letters if char not in seen]
    
    for i, missing_char in zip(duplicates, missing):
        child[i] = missing_char
    
    return child

def mutation(population, mutation_rate=0.1):
    """
    Apply swap mutation to population
    
    Args:
        population: List of keyboard layouts
        mutation_rate: Probability of mutation for each individual
    
    Returns:
        list: Mutated population
    """
    mutated_population = []
    
    for layout in population:
        new_layout = layout.copy()
        if np.random.rand() < mutation_rate:
            pos1, pos2 = np.random.choice(30, 2, replace=False)
            new_layout[pos1], new_layout[pos2] = new_layout[pos2], new_layout[pos1]
        
        mutated_population.append(new_layout)
    
    return mutated_population

def replacement(old_population, new_offspring):
    """
    Replace old population with new offspring
    
    Args:
        old_population: Current population (sorted by fitness, best first)
        new_offspring: New offspring from crossover and mutation
        replacement_method: "generational" (replace all) or "elitist" (keep best)
    
    Returns:
        list: New population
    """

    
    elite_size = len(old_population) // 2
    elite = old_population[:elite_size]  
    new_population = elite + new_offspring[:len(old_population) - elite_size]
    return new_population
    

def tournament_selection(population_with_fitness, tournament_size=3):
    """
    Perform tournament selection to select one parent
    
    Args:
        population_with_fitness: List of (layout, fitness) tuples
        tournament_size: Number of individuals in each tournament
    
    Returns:
        Selected parent layout
    """
    tournament = random.sample(population_with_fitness, k=tournament_size)
    winner = min(tournament, key=lambda item: item[1])  # Winner has lowest fitness
    return winner[0]  # Return just the layout


# def run_genetic_algorithm(text_file, pop_size=100, generations=50, elite_size=10, tournament_size=3, crossover_rate=0.8, mutation_rate=0.1, use_known_distributions=True):
#     """Run the genetic algorithm to optimize keyboard layouts"""
    
#     print("--- Starting Genetic Algorithm (with Tournament Selection) ---")
#     text = load_text_from_file(text_file)
#     if text is None:
#         print("Could not load text. Aborting.")
#         return None, None
        
#     print(f"Loaded {len(text)} characters from '{text_file}'.")
    
#     population = init_population(pop_size, use_known_distributions)
#     print(f"Initialized population of {pop_size} layouts.")

#     history = {"best_fitness": [], "avg_fitness": []}
#     best_layout_ever = None
#     best_fitness_ever = float('inf')

#     # Generational Loop
#     for gen in range(generations):
#         # Calculate fitness for current population
#         fitness_scores = fitness_function(population, text)
        
#         # Create population with fitness pairs for easier handling
#         population_with_fitness = list(zip(population, fitness_scores))
#         population_with_fitness.sort(key=lambda item: item[1])  # Sort by fitness (best first)
        
#         # Extract sorted population and fitness for reporting
#         sorted_population = [item[0] for item in population_with_fitness]
#         sorted_fitness = [item[1] for item in population_with_fitness]
        
#         # Track best and average fitness
#         current_best_fitness = sorted_fitness[0]
#         current_avg_fitness = np.mean(sorted_fitness)
#         history["best_fitness"].append(current_best_fitness)
#         history["avg_fitness"].append(current_avg_fitness)

#         if current_best_fitness < best_fitness_ever:
#             best_fitness_ever = current_best_fitness
#             best_layout_ever = sorted_population[0].copy()

#         print(f"Generation {gen+1:02}/{generations} -> Best Fitness: {current_best_fitness:,.2f}, Avg Fitness: {current_avg_fitness:,.2f}")

#         # Create next generation
#         next_generation = []

#         elite = [layout.copy() for layout in sorted_population[:elite_size]]
#         next_generation.extend(elite)
        
#         num_offspring = pop_size - elite_size
        
#         for _ in range(num_offspring):
#             parent1 = tournament_selection(population_with_fitness, tournament_size)
#             parent2 = tournament_selection(population_with_fitness, tournament_size)

#             # Crossover
#             if random.random() < crossover_rate:
#                 child = crossover(parent1, parent2)
#             else:
#                 child = parent1.copy()  
            
#             next_generation.append(child)

#         next_generation = mutation(next_generation, mutation_rate)
        
#         # Update population for next iteration
#         population = next_generation

#     print("\n--- Genetic Algorithm Finished ---")
#     return best_layout_ever, history

def run_genetic_algorithm(text_file, pop_size=100, generations=50, elite_size=10, 
                                 tournament_size=3, crossover_rate=0.8, mutation_rate=0.1, 
                                 use_known_distributions=True):
    """
    Versión actualizada del algoritmo genético con crossover de 2 hijos
    """
    print("--- Starting Genetic Algorithm (with Two-Child Crossover) ---")
    
    # ... código de inicialización igual ...
    text = load_text_from_file(text_file)
    if text is None:
        print("Could not load text. Aborting.")
        return None, None
        
    print(f"Loaded {len(text)} characters from '{text_file}'.")
    
    population = init_population(pop_size, use_known_distributions)
    print(f"Initialized population of {pop_size} layouts.")

    history = {"best_fitness": [], "avg_fitness": []}
    best_layout_ever = None
    best_fitness_ever = float('inf')

    # Generational Loop
    for gen in range(generations):
        # Calculate fitness for current population
        fitness_scores = fitness_function(population, text)
        
        # Create population with fitness pairs for easier handling
        population_with_fitness = list(zip(population, fitness_scores))
        population_with_fitness.sort(key=lambda item: item[1])  # Sort by fitness (best first)
        
        # Extract sorted population and fitness for reporting
        sorted_population = [item[0] for item in population_with_fitness]
        sorted_fitness = [item[1] for item in population_with_fitness]
        
        # Track best and average fitness
        current_best_fitness = sorted_fitness[0]
        current_avg_fitness = np.mean(sorted_fitness)
        history["best_fitness"].append(current_best_fitness)
        history["avg_fitness"].append(current_avg_fitness)

        if current_best_fitness < best_fitness_ever:
            best_fitness_ever = current_best_fitness
            best_layout_ever = sorted_population[0].copy()

        print(f"Generation {gen+1:02}/{generations} -> Best Fitness: {current_best_fitness:,.2f}, Avg Fitness: {current_avg_fitness:,.2f}")

        # Create next generation
        next_generation = []

        elite = [layout.copy() for layout in sorted_population[:elite_size]]
        next_generation.extend(elite)
        
        num_offspring_needed = pop_size - elite_size
        num_pairs = num_offspring_needed // 2  
        
        offspring = []
        
        for _ in range(num_pairs):
            parent1 = tournament_selection(population_with_fitness, tournament_size)
            parent2 = tournament_selection(population_with_fitness, tournament_size)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)  
            else:
                child1, child2 = parent1.copy(), parent2.copy()  
            
            # Añadir ambos hijos
            offspring.extend([child1, child2])
        
        # Si necesitamos un hijo impar adicional
        if len(offspring) < num_offspring_needed:
            parent1 = tournament_selection(population_with_fitness, tournament_size)
            parent2 = tournament_selection(population_with_fitness, tournament_size)
            
            if random.random() < crossover_rate:
                child1, _ = crossover(parent1, parent2)  # Solo usar el primer hijo
            else:
                child1 = parent1.copy()
            
            offspring.append(child1)
        
        # Combinar elite + offspring (tomar solo los necesarios)
        next_generation.extend(offspring[:num_offspring_needed])
        
        # ===== FIN SECCIÓN ACTUALIZADA =====

        # Apply mutation to the entire next generation (including elites)
        next_generation = mutation(next_generation, mutation_rate)
        
        # Update population for next iteration
        population = next_generation

    print("\n--- Genetic Algorithm Finished ---")
    return best_layout_ever, history 


def print_keyboard(layout, name="Keyboard"):
    """Print keyboard in 3x10 format"""
    print(f"\n{name}:")
    for row in range(3):
        row_keys = layout[row*10:(row+1)*10]
        print(" ".join(f"{key:>2}" for key in row_keys))
    
def test_genetic_operators():
    """Test the genetic algorithm components"""
    population = init_population(10, known_distributions=True)
    sample_text = "the quick brown fox jumps over the lazy dog"
    
    print("=== TESTING GENETIC OPERATORS ===")
    
    # Test selection
    selected, fitness = selection(population, sample_text, "elitist")
    print("Selection results (fitness scores):")
    for i, f in enumerate(fitness[:5]):
        print(f"  Individual {i}: {f:.2f}")
    
    # Test crossover
    child = crossover(selected[0], selected[1])
    print(f"\nCrossover result (first 10 keys): {child[:10]}")
    
    # Verify crossover produces valid permutation
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ',', '.', ';', "'"]
    if set(child) == set(letters) and len(child) == 30:
        print("✓ Crossover produces valid permutation")
    else:
        print("✗ Crossover error - invalid permutation")
    
    # Test mutation
    mutated = mutation([child], mutation_rate=1.0)
    print(f"Mutation result (first 10 keys): {mutated[0][:10]}")
    
    print("\nGenetic operators working correctly!")
      
def test_system():
    """Test the optimized system"""
    population = init_population(4, known_distributions=True)
    sample_text = "the quick brown fox jumps over the lazy dog and then runs away quickly so that it can hide from the hunter in the forest and avoid being caught eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

    
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


def plot_history(history, generations):
    """
    Plots the best and average fitness over generations.

    Args:
        history (dict): Dictionary with 'best_fitness' and 'avg_fitness' lists.
        generations (int): Total number of generations for the x-axis.
    """
    if not history or not history['best_fitness']:
        print("No history to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(history['best_fitness'], label='Best Fitness', color='blue', linewidth=2)
    plt.plot(history['avg_fitness'], label='Average Fitness', color='cyan', linestyle='--')
    plt.title('Keyboard Layout Fitness Evolution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score (Lower is Better)')
    plt.xticks(np.arange(0, generations, step=max(1, generations//10)))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    POPULATION_SIZE = 20
    GENERATIONS = 30
    ELITE_SIZE = 10           # Keep the top 10%
    TOURNAMENT_SIZE = 3      # Standard tournament size
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.25
    TEXT_FILE = 'data/moby_dick_cln.txt'  

    best_layout, evolution_history = run_genetic_algorithm(
        text_file=TEXT_FILE,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_size=ELITE_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        use_known_distributions=True
    )

    if best_layout:
        qwerty = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', "'"]
        text_content = load_text_from_file(TEXT_FILE)
        qwerty_fitness = fitness_function([qwerty], text_content)[0]

        print("\n--- Final Results ---")
        
        print(f"\nQWERTY Fitness Score: {qwerty_fitness:,.2f}")
        print_keyboard(qwerty, "QWERTY Layout")

        best_fitness = evolution_history['best_fitness'][-1]
        print(f"\nBest Evolved Layout Fitness Score: {best_fitness:,.2f}")
        print_keyboard(best_layout, "Best Evolved Layout")

        #plot_history(evolution_history, GENERATIONS)