import pygmo as pg
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

"""
This script performs multi-objective optimization on bicycle routes using pygmo's NSGA-II and MOEA/D algorithms.
It loads a dataset of routes, defines a custom optimization problem, runs the algorithms, and visualizes the results.
"""
# ==============================================================================================
# ======================================== Load dataset ========================================
# ==============================================================================================
df = pd.read_csv("sampled_routes_with_metrics.csv")
np.random.seed(42)
pg.set_global_rng_seed(42)
edges = {}
dataset_points = []

# Load and process the dataset, extracting route metrics and edge information.
for _, row in df.iterrows():
    path = ast.literal_eval(row["path"])
    distances = ast.literal_eval(row["raw_distance"])
    beauties = ast.literal_eval(row["raw_beauty"])
    roughnesses = ast.literal_eval(row["raw_roughness"])
    safeties = ast.literal_eval(row["raw_safety"])
    slopes = ast.literal_eval(row["raw_slope"])

    num_edges = len(path) - 1
    total_distance = sum(distances)
    total_comfort = (sum(beauties) + sum(roughnesses) + sum(safeties) + sum(slopes)) / (
        4 * num_edges
    )
    dataset_points.append((float(total_distance), total_comfort))

    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edge = tuple(sorted([a, b]))
        comfort = beauties[i] + roughnesses[i] + safeties[i] + slopes[i]
        edges[edge] = {"distance": distances[i], "comfort": comfort}


# ==================================================================================================
# ===================================== Define optimization problem ================================
# ==================================================================================================
class RoutePermutationProblem:
    """
    A pygmo-compatible problem for optimizing bicycle routes with variable length.

    Attributes:
        all_nodes (list): List of intermediate node indices.
        start (int): Start node index.
        end (int): End node index.
        min_len (int): Minimum number of intermediate nodes in a route.
        max_len (int): Maximum number of intermediate nodes in a route.
        route_length (int): Fixed encoding length for the optimizer.
    """

    def __init__(self, all_nodes, start=0, end=19, min_len=7, max_len=10):
        """
        Initialize the route optimization problem.

        Args:
            all_nodes (list): List of intermediate node indices.
            start (int): Start node index.
            end (int): End node index.
            min_len (int): Minimum route length.
            max_len (int): Maximum route length.
        """
        self.all_nodes = all_nodes
        self.start = start
        self.end = end
        self.min_len = min_len
        self.max_len = max_len
        self.route_length = max_len

    def fitness(self, x):
        """
        Compute the objectives for a candidate route.

        Args:
            x (array-like): Encoded route as a vector of indices.

        Returns:
            list: [total_distance, -average_comfort]
        """
        x_clean = []
        seen = set()
        for i in x:
            idx = int(round(i))
            if 0 <= idx < len(self.all_nodes) and idx not in seen:
                x_clean.append(self.all_nodes[idx])
                seen.add(idx)
            if len(x_clean) >= self.max_len:
                break

        route_len = np.random.randint(self.min_len, self.max_len + 1)
        x_short = x_clean[:route_len]
        route = [self.start] + x_short + [self.end]

        dist = 0
        comfort = 0
        for a, b in zip(route[:-1], route[1:]):
            edge = tuple(sorted([a, b]))
            if edge in edges:
                dist += edges[edge]["distance"]
                comfort += edges[edge]["comfort"]
            else:
                dist += 9999

        num_edges = len(route) - 1
        avg_comfort = comfort / (4 * num_edges) if num_edges > 0 else 0
        return [dist, -avg_comfort]

    def get_bounds(self):
        """
        Return the bounds for the optimization variables.

        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        return (
            np.zeros(self.route_length),
            np.full(self.route_length, len(self.all_nodes) - 1),
        )

    def get_nobj(self):
        """
        Return the number of objectives.

        Returns:
            int: Number of objectives (2).
        """
        return 2

    def get_nix(self):
        """
        Return the number of decision variables.

        Returns:
            int: Number of variables.
        """
        return self.route_length

    def get_name(self):
        """
        Return the name of the problem.

        Returns:
            str: Problem name.
        """
        return "Variable-Length Route Optimization"


def run_algorithm(algo_name):
    """
    Run a multi-objective evolutionary algorithm on the problem.

    Args:
        algo_name (str): Name of the algorithm ('nsga2' or 'moea').

    Returns:
        pygmo.population: Evolved population.
    """
    if algo_name == "nsga2":
        algo = pg.algorithm(pg.nsga2(gen=500))
    elif algo_name == "moead":
        algo = pg.algorithm(pg.moead(gen=500))
    else:
        raise ValueError("Unsupported algorithm")
    algo.set_seed(42)

    pop = pg.population(prob, size=1000)
    pop = algo.evolve(pop)
    return pop


# =======================================================================================================
# ==================================== Optimization =====================================================
# =======================================================================================================
intermediate_nodes = list(range(1, 19))
prob = pg.problem(RoutePermutationProblem(intermediate_nodes, min_len=7, max_len=10))

# NSGA-II and moead
pop_nsga2 = run_algorithm("nsga2")
pop_moead = run_algorithm("moead")


def extract_pareto_front(f_vals):
    """
    Extracts and sorts the Pareto front from objective values.
    Returns: (indices, sorted_front)
    """
    indices = pg.fast_non_dominated_sorting(f_vals)[0][0]
    front = f_vals[indices]
    front = front[np.argsort(front[:, 0])]
    return indices, front


def plot_solutions_and_fronts(
    dataset_np, f_vals_nsga2, f_vals_moead, pareto_f_nsga2, pareto_f_moead
):
    plt.figure(figsize=(12, 8))
    # Dataset routes
    plt.scatter(
        dataset_np[:, 0],
        dataset_np[:, 1],
        color="gray",
        alpha=0.4,
        label="Dataset Routes",
    )
    # Solutions of algorithms
    plt.scatter(
        f_vals_nsga2[:, 0],
        -f_vals_nsga2[:, 1],
        color="blue",
        alpha=0.6,
        marker="o",
        label="NSGA-II Solutions",
    )
    plt.scatter(
        f_vals_moead[:, 0],
        -f_vals_moead[:, 1],
        color="red",
        alpha=0.6,
        marker="s",
        label="MOEA/D Solutions",
    )
    # Pareto fronts
    plt.plot(
        pareto_f_nsga2[:, 0],
        -pareto_f_nsga2[:, 1],
        color="blue",
        linewidth=2,
        label="Pareto Front (NSGA-II)",
    )
    plt.plot(
        pareto_f_moead[:, 0],
        -pareto_f_moead[:, 1],
        color="red",
        linewidth=2,
        label="Pareto Front (MOEA/D)",
    )
    plt.xlabel("Total Distance (minimize)")
    plt.ylabel("Total Comfort (maximize)")
    plt.title("Comparison of NSGA-II and MOEA/D Results")  # corrected spelling
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Get objective values for both algorithms
f_vals_nsga2 = np.array(pop_nsga2.get_f())
f_vals_moead = np.array(pop_moead.get_f())

# Compute Pareto fronts for both algorithms
pareto_indices_nsga2, pareto_f_nsga2 = extract_pareto_front(f_vals_nsga2)
pareto_indices_moead, pareto_f_moead = extract_pareto_front(f_vals_moead)

dataset_np = np.array(dataset_points)

# Plot the results
plot_solutions_and_fronts(
    dataset_np, f_vals_nsga2, f_vals_moead, pareto_f_nsga2, pareto_f_moead
)

print("\nAlgorithm Comparison Statistics:")
print("******************************************")
print("******************************************")
print("******************************************")

for algo_name, f_vals in [
    ("NSGA-II", f_vals_nsga2),
    ("MOEA/D", f_vals_moead),
]:  # corrected spelling
    print(f"\n{algo_name}:")
    print(f"Number of solutions: {len(f_vals)}")
    print(f"Distance range: {f_vals[:, 0].min():.2f} - {f_vals[:, 0].max():.2f}")
    print(f"Comfort range: {-f_vals[:, 1].min():.2f} - {-f_vals[:, 1].max():.2f}")

    # Calculate how many solutions match dataset routes
    evolved_set = set(tuple(np.round([d, -c], 2)) for d, c in f_vals)
    dataset_set = set(tuple(np.round([d, c], 2)) for d, c in dataset_points)
    overlap = evolved_set & dataset_set
    print(f"Number of solutions matching dataset routes: {len(overlap)}")


plt.figure(figsize=(8, 6))
plt.scatter(dataset_np[:, 0], dataset_np[:, 1], color="gray", alpha=0.6)
plt.xlim(400, 1400)
plt.ylim(2.4, 3.8)
plt.xlabel("Total Distance (minimize)")
plt.ylabel("Average Comfort (maximize)")
plt.title("Dataset Bicycle Routes")
plt.grid(True)
plt.tight_layout()
plt.show()


def decode_route(x, all_nodes, start=0, end=19, max_len=10):
    """
    Decode a vector of indices into a route.

    Args:
        x (array-like): Encoded route as a vector of indices.
        all_nodes (list): List of intermediate node indices.
        start (int): Start node index.
        end (int): End node index.
        max_len (int): Maximum number of intermediate nodes.

    Returns:
        list: Decoded route as a list of node indices.
    """
    x_clean = []
    seen = set()
    for i in x:
        idx = int(round(i))
        if 0 <= idx < len(all_nodes) and idx not in seen:
            x_clean.append(all_nodes[idx])
            seen.add(idx)
        if len(x_clean) >= max_len:
            break
    return [start] + x_clean + [end]


# ==================================================================================================
# ================================== Extract Pareto front routes ===================================
# ==================================================================================================
# For each solution on the NSGA-II Pareto front, decode the route and collect its details.
# 1. to check if there is no overlap with the dataset routes.
# 2. to get the decode the nodes from the solution vector with its corresponding metrics to show on the plot.
X_nsga2 = pop_nsga2.get_x()
F_nsga2 = pop_nsga2.get_f()
pareto_indices_nsga2 = pg.fast_non_dominated_sorting(F_nsga2)[0][0]

X_moead = pop_moead.get_x()
F_moead = pop_moead.get_f()
pareto_indices_moead = pg.fast_non_dominated_sorting(F_moead)[0][0]

records = []

# NSGA-II Pareto front
for i in pareto_indices_nsga2:
    route = decode_route(X_nsga2[i], intermediate_nodes)
    records.append(
        {
            "source": "NSGA-II",
            "route": route,
            "distance": float(F_nsga2[i][0]),
            "avg_comfort": float(-F_nsga2[i][1]),
        }
    )

print(f"records: {records}")
