# Bicycle Route Optimization

This project uses evolutionary algorithms to find optimal bicycle routes based on distance and comfort metrics.

## Overview

The `main.py` script performs the following steps:

1.  **Load and Process Data:** Reads our generated dataset of bicycle routes and extracts edge information (distance and comfort attributes) to build a graph representation.
2.  **Define Optimization Problem:** Sets up a multi-objective optimization problem using the PyGMO library, defining the fitness function (calculating distance and comfort for a given route) and the decision space.
3.  **Run Evolutionary Algorithms:** Applies two different multi-objective evolutionary algorithms (NSGA-II and MOEA/D) to find a set of optimal routes that represent trade-offs between minimizing distance and maximizing comfort.
4.  **Analyze and Visualize Results:** Extracts the Pareto front from the solutions found by each algorithm and visualizes them in a plot comparing the performance of the algorithms against the original dataset.

## Code Structure Explanation

-   **Dataset Loading and Processing:** The initial part of the script reads `sampled_routes_with_metrics.csv` and populates the `edges` dictionary with edge data (distance and combined comfort score) and `dataset_points` with the objective values of the original dataset routes.
-   **`RoutePermutationProblem` Class:** Defines the optimization problem for PyGMO. The `fitness` method takes a candidate solution (a vector `x` representing a sequence of intermediate nodes) and returns the objective values (total distance and the negative of average comfort) for the corresponding route.
-   **`run_algorithm` Function:** A helper function.