import random
import networkx as nx
import pandas as pd

random.seed(42)


def create_graph(N=20, E=140, verbose=False):
    """
    Create a synthetic bike network graph with comfort attributes.
    Returns: the graph object (networkx.Graph)
    """
    G = nx.Graph()
    nodes = list(range(N))
    G.add_nodes_from(nodes)

    while G.number_of_edges() < E:
        i, j = random.sample(nodes, 2)
        if G.has_edge(i, j):
            continue

        distance = random.randint(50, 200)
        raw_beauty = random.randint(1, 5)
        raw_roughness = random.randint(1, 5)
        raw_safety = random.randint(1, 5)
        raw_slope = random.randint(1, 5)

        comfort_beauty = raw_beauty
        comfort_roughness = raw_roughness
        comfort_safety = 6 - raw_safety
        comfort_slope = 6 - raw_slope

        G.add_edge(
            i,
            j,
            distance=distance,
            raw_beauty=raw_beauty,
            raw_roughness=raw_roughness,
            raw_safety=raw_safety,
            raw_slope=raw_slope,
            comfort_beauty=comfort_beauty,
            comfort_roughness=comfort_roughness,
            comfort_safety=comfort_safety,
            comfort_slope=comfort_slope,
        )

    df = graph_to_df(G)
    df.to_csv("synthetic_bike_routes.csv", index=False)
    if verbose:
        print(df.head())
        print(df.shape)
    return G


def graph_to_df(G):
    """
    Convert a networkx graph to a pandas DataFrame.
    G: networkx graph
    returns: pandas DataFrame
    """
    edges_data = []
    for u, v, attr in G.edges(data=True):
        edges_data.append({"node1": u, "node2": v, **attr})
    return pd.DataFrame(edges_data)


if __name__ == "__main__":
    create_graph(N=20, E=60, verbose=True)
