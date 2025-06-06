import pandas as pd
import networkx as nx
from networkx import all_simple_paths
import random
import numpy as np


def load_graph_from_csv(filepath):
    df = pd.read_csv(filepath)
    G = nx.from_pandas_edgelist(
        df,
        source="node1",
        target="node2",
        edge_attr=[
            "distance",
            "comfort_beauty",
            "comfort_roughness",
            "comfort_safety",
            "comfort_slope",
        ],
    )
    return G


def find_paths(G, min_length=8, max_length=11):
    """
    Find all paths between node 0 and 19 with length between 7 and 10 nodes.
    Saves results to paths_with_8_nodes.csv
    """
    paths_min_length_max_length_nodes = [
        path
        for path in all_simple_paths(G, source=0, target=19, cutoff=max_length)
        if min_length - 1 <= len(path) <= max_length - 1
    ]

    return paths_min_length_max_length_nodes


def create_paths(G):
    """
    Create routes from paths in df
    """
    paths_min_length_max_length_nodes = find_paths()
    pd.DataFrame(paths_min_length_max_length_nodes).to_csv(
        f"paths_min_length_7_max_length_10_nodes.csv", index=False
    )
    # df_whole = pd.read_csv("paths_min_length_8_max_length_11_nodes.csv")
    # paths = df_whole.sample(n=300, random_state=42)
    # return paths


def evaluate_path(G, path, weights):
    total_distance = 0
    beauty, roughness, safety, slope, distances = [], [], [], [], []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = G[u][v]
        d = edge["distance"]
        distances.append(d)
        total_distance += d
        beauty.append(edge["comfort_beauty"])
        roughness.append(edge["comfort_roughness"])
        safety.append(edge["comfort_safety"])
        slope.append(edge["comfort_slope"])

    comfort_safety = [6 - s for s in safety]
    comfort_slope = [6 - s for s in slope]

    return {
        "path": path,
        "distance": total_distance,
        "comfort_beauty": sum(beauty),
        "comfort_roughness": sum(roughness),
        "comfort_safety": sum(comfort_safety),
        "comfort_slope": sum(comfort_slope),
        "avg_comfort": np.mean(
            [
                np.mean(beauty),
                np.mean(roughness),
                np.mean(comfort_safety),
                np.mean(comfort_slope),
            ]
        ),
        "raw_distance": distances,
        "raw_beauty": beauty,
        "raw_roughness": roughness,
        "raw_safety": safety,
        "raw_slope": slope,
    }


if __name__ == "__main__":
    random.seed(42)
    G = load_graph_from_csv("synthetic_bike_routes.csv")
    all_paths = find_paths(G)
    print(f"Found {len(all_paths)} valid paths.")
    sampled_paths = random.sample(all_paths, min(300, len(all_paths)))
    pd.DataFrame(sampled_paths).to_csv("sampled_paths.csv", index=False)

    # User preferences
    weights = {"beauty": 0.2, "roughness": 0.3, "safety": 0.35, "slope": 0.15}
    metrics = [evaluate_path(G, path, weights) for path in sampled_paths]
    df = pd.DataFrame(metrics)
    df.to_csv("sampled_routes_with_raw_metrics.csv", index=False)

    print(f"✅ Saved routes with metrics.")
