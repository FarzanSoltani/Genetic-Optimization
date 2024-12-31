# Genetic-Optimization

This repository contains a Python implementation of a **Genetic Algorithm (GA)** designed to optimize numerical functions. The algorithm uses evolutionary principles like mutation, recombination, and selection to improve solutions iteratively.

---

## Features

- **Parallel Processing**: Accelerates evaluations using Python's multiprocessing module.
- **Customizable Parameters**: Adjust population size, mutation rate, tournament size, and more.
- **Fitness Evaluation**: Supports user-defined objective functions for flexible optimization tasks.
- **Result Tracking**: Logs the best parameters and fitness values over generations.

---

## How It Works

The Genetic Algorithm follows these steps:
1. **Population Initialization**: Randomly generates individuals within defined bounds.
2. **Fitness Evaluation**: Scores individuals using a user-defined function.
3. **Selection**: Selects the best parents through tournament selection.
4. **Recombination**: Combines parents to produce offspring.
5. **Mutation**: Alters offspring parameters based on mutation rates.
6. **Iteration**: Repeats the process for a specified number of generations.

---

## Prerequisites

- **Python 3.x**
- Required Libraries:
  - `numpy`
  - `multiprocessing`
  - `datetime`

---

## Usage

### Step 1: Import the Class
```python
from genetic_strategy import GeneticStrategy
```
### Step 2: Define Your Objective Function
```def objective_function(params):
    return -sum(x**2 for x in params)  # Example: minimize the sum of squares
```
### Step 3: Initialize the Algorithm
```ga = GeneticStrategy(
    function=objective_function,
    num_generations=100,
    population_size=50,
    mutation_rate=0.1,
    tournament_size=0.2,
    bound=[-10, 10],  # Parameter bounds
    dimension=7,      # Number of variables
    n_worker=4,
    database="database_name"
)
```
### Step 4: Run the Optimization
```
ga.run()
print("Best Result:", ga.best_resault)
print("Best Parameters:", ga.best_variable)
```

---

## Example Output

```
Iteration: 0
Iteration: 1
...
Iteration: 99
Best Result: -0.001
Best Parameters: [0.01, 0.03, 0.02, ...]
```

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
