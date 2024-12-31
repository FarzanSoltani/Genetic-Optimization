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
