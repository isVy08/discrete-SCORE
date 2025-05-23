import numpy as np

def is_valid_topological_order(adj_matrix, ordering):
    """
    Check if the given ordering is a valid topological ordering of the graph.

    Parameters:
        adj_matrix (np.ndarray): Binary adjacency matrix of the graph (shape: n x n).
                                 adj_matrix[i, j] = 1 indicates an edge from node i to node j.
        ordering (list): List of nodes representing the proposed topological ordering.

    Returns:
        bool: True if the ordering is valid, False otherwise.
    """
    # Create a maporderingng of node to its position in the ordering
    position = {node: idx for idx, node in enumerate(ordering)}
    
    # Iterate over all edges in the adjacency matrix
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:  # There's an edge from node i to node j
                # Ensure i appears before j in the ordering
                if position[i] > position[j]:
                    return False  # Invalid topological order

    return True  # All edges respect the topological order

def D_top(adj_matrix, ordering):
    err = 0
    for i in range(len(ordering)):
        err += adj_matrix[ordering[i+1:], ordering[i]].sum()
    return err


def generate_forbidden_links(ordered_vertices):
    """
    list of index of nodes, from leaf to root
    """
    links = {}
    n = len(ordered_vertices)
    for i in range(n): 
        links[str(ordered_vertices[i])] = [str(j) for j in ordered_vertices[i+1:]]
    return links

def generate_accepted_links(ordered_vertices):
    """
    list of index of nodes, from leaf to root
    """
    links = {}
    for i in range(len(ordered_vertices)): 
        links[str(ordered_vertices[i])] = [str(j) for j in ordered_vertices[:i]]
    return links

def full_DAG(top_order):
    """
    list of index of nodes, from root to leaf
    """
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order
