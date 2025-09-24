import numpy as np
import random
from random import randint, shuffle

def init_population(pop_size, known_distributions=False):

    # Our complete character set (30 characters)
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
        print(new_individual)
    return keyboards


def print_keyboard_layout(layout, name="Keyboard"):

    print(f"\n{name} Layout:")
    print("+" + "-" * 29 + "+")
    for row in range(3):
        row_str = "|"
        for col in range(10):
            pos = row * 10 + col
            key = layout[pos] if layout[pos] != "'" else "'"  
            row_str += f" {key} "
        row_str += "|"
        print(row_str)
    print("+" + "-" * 29 + "+")   
    
          
# Test the function
if __name__ == "__main__":
    # Test with known distributions
    population = init_population(10, known_distributions=True)
    
    # Print all known layouts
    layout_names = ['QWERTY', 'Dvorak', 'QWERTZ', 'Colemak']
    for i in range(min(4, len(population))):
        print_keyboard_layout(population[i], layout_names[i])
    
    print(f"\nTotal population size: {len(population)}")
    print(f"Each layout has {len(population[0])} keys")
    