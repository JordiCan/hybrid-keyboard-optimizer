# Hybrid Keyboard Optimizer

A hybrid optimization system for keyboard layout design that combines genetic algorithms and simulated annealing to minimize typing fatigue and maximize efficiency.

## Motivation

The QWERTY layout dates back to 1870 and was designed to prevent mechanical jams in typewriters, not to optimize human ergonomics. This project uses algorithmic optimization techniques to explore alternative layouts based on modern typing patterns and character frequency analysis.

## Methodology

The system implements two metaheuristic algorithms:

**Genetic Algorithms (GA)**
Generates populations of keyboard layouts, selects the most efficient ones through a fitness function, and produces new generations via crossover and mutation operators.

**Simulated Annealing (SA)**
Refines candidate solutions through a gradual cooling process that temporarily accepts suboptimal configurations to escape local minima.

**Hybrid Approach**
The combination of both methods enables broad exploration of the solution space (GA) followed by local refinement (SA), yielding better solutions than either algorithm achieves individually.

## Fitness Function

The system evaluates each layout according to multiple criteria:

- **Finger travel distance**: Total distance traveled by fingers
- **Hand alternation**: Frequency of switching between left and right hand
- **Same-finger penalty**: Consecutive use of the same finger
- **Home row usage**: Percentage of keystrokes on the main row
- **Bigram optimization**: Efficient placement of frequent letter pairs
- **Hand balance**: Equal workload distribution

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: numpy, matplotlib, jupyter

## Usage

Run the Jupyter notebooks:

```bash
jupyter notebook
```

Available files:
- `cross_genetic_algorithm.ipynb` - Genetic algorithm implementation
- `cross_simulated_annealing.ipynb` - Simulated annealing implementation

The notebooks include complete implementations, optimization process visualizations, and comparisons with established layouts (QWERTY, Dvorak, Colemak).

## Visualizations

The project generates:
- Heatmaps of finger usage
- Algorithm convergence plots
- Performance comparisons
- Finger travel distance metrics

## Considerations

- The optimal layout depends on language, content type, and individual preferences
- Small efficiency improvements may require significant layout changes
- Muscle memory represents a practical obstacle for adopting new layouts
- Different use contexts (programming vs. prose writing) may benefit from distinct layouts

## Academic Application

This project is useful for:
- Study of metaheuristic optimization algorithms
- Research in human-computer interaction
- Ergonomic design analysis
- Practical application of computational intelligence techniques

## Contributing

Contributions are welcome. Areas of interest:
- Fitness function improvements
- New optimization strategies
- Multi-language support
- Experimental validation with real users

