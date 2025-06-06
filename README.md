# Bicycle Route Optimization

This project uses evolutionary algorithms to find optimal bicycle routes based on distance and comfort metrics.

## Overview

This project consists of two main scripts:

- `generate_data.py`: Generates the dataset of bicycle routes and their metrics.
- `prepare_dataset.py`: Samples 300 pre-computed routes from the previouse step to make the computation easier when running evolutionary algorithms. We work with this sampled dataset throught the project.
- `main.py`: Performs multi-objective optimization on the generated dataset using evolutionary algorithms (NSGA-II and MOEA/D) and visualizes the results.

### Workflow

1. **Generate Dataset**
   - Run `generate_data.py` to create the dataset file `synthetic_bike_routes`.
   - Run `prepare_dataset.py` to sample 300 routes from `synthetic_bike_routes` with 4 comfort and distance metrics and creates the sampled dataset file `sampled_routes_with_metrics.csv`. 

2. **Run Optimization and Visualization**
   - Run `main.py` to:
     - Load and process the dataset.
     - Define a custom multi-objective optimization problem using PyGMO.
     - Run NSGA-II and MOEA/D algorithms to find optimal routes.
     - Visualize and compare the results, including Pareto fronts and dataset routes.
     - Print statistics and details about the solutions.

## Installation and Running the Project

### 1. Clone the Repository

```sh
git clone git@github.com:NastaranMO/MODA-A2.git
cd MODA-A2
```

### 2. Install Dependencies

This project requires Python 3.8+ and the following packages:
- `pygmo`
- `pandas`
- `numpy`
- `matplotlib`

You can install the dependencies using pip:

```bash
pip install pygmo pandas numpy matplotlib
```

> **Note:** On macOS, you may need to install `pygmo` via conda or brew if pip fails. See [PyGMO installation instructions](https://esa.github.io/pygmo2/install.html) for details.

### 3. Generate the Dataset

```bash
python generate_data.py
```

This will create `sampled_routes_with_metrics.csv` in the project directory.

### 4. Run the Optimization and Visualization

```bash
python main.py
```

This will:
- Run the optimization algorithms.
- Display plots comparing the solutions and Pareto fronts.
- Print statistics about the solutions in the terminal.

## Code Structure

- **Dataset Loading and Processing:** Reads `sampled_routes_with_metrics.csv` and extracts edge and route metrics.
- **`RoutePermutationProblem` Class:** Defines the optimization problem for PyGMO, including the fitness function and variable bounds.
- **Run Evolutionary Algorithms:** Applies two different multi-objective evolutionary algorithms (`NSGA-II` and `MOEA/D`) to find a set of optimal routes that represent trade-offs between minimizing distance and maximizing comfort.
- **Visualization:** Plots the dataset, algorithm solutions, and Pareto fronts for comparison.
- **Pareto Front Extraction:** Extracts and decodes the Pareto-optimal routes for further analysis.

## Output
- Plots comparing dataset routes, NSGA-II, and MOEA/D solutions.
- Printed statistics about the number and quality of solutions.
- Example output of decoded Pareto front routes in the terminal.

---

For any issues with dependencies or running the scripts, please refer to the official documentation of each package or open an issue in this repository.