import random, os
import numpy as np
import networkx as nx
import igraph as ig
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from utils.io import load_pickle, write_pickle

class SyntheticDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
    """

    def __init__(self, root, config_code, num_samples, num_nodes, max_cardinality, min_cardinality, graph_type, degree):

        self.data_path = f'./{root}/{config_code}.pickle'
        self.num_samples = num_samples 
        self.num_nodes = num_nodes 
        self.max_cardinality = max_cardinality
        self.min_cardinality = min_cardinality
        self.graph_type = graph_type
        self.degree = degree
        self.config_code = config_code

        os.makedirs(root, exist_ok=True)

        if os.path.isfile(self.data_path):
            print('Loading data ...')
            self.X, self.B_bin, self.model, self.cards = load_pickle(self.data_path)
        else:
            print('Generating and Saving data ...')
            self._setup()
            self.X = self.X.loc[:, list(range(self.num_nodes))]
            package = (self.X, self.B_bin, self.model, self.cards)
            
            write_pickle(package, self.data_path)

    def _setup(self):
        self.cards = [random.choice(range(self.min_cardinality, self.max_cardinality + 1)) for _ in range(self.num_nodes)]
        self.B_bin = SyntheticDataset.simulate_random_dag(self.num_nodes, self.degree, self.graph_type)
        self.G = nx.DiGraph(self.B_bin)

        
        self.model = BayesianNetwork() # must check all nodes are connected.
        self.model.add_nodes_from(list(range(self.num_nodes)))
        self.model.add_edges_from(list(self.G.edges))
        
        
        cpds = self._simulate_cpds()
        
        for cpd in cpds:
            self.model.add_cpds(cpd)
            
        if not self.model.check_model():
            raise ValueError("The Bayesian Network model is invalid. Please check your CPDs.")

        # Simulate data samples from the Bayesian network
        inference = BayesianModelSampling(self.model)
        self.X = inference.forward_sample(size=self.num_samples)
    
    def _simulate_cpds(self): 
        all_cpds = []
        for variable in range(self.num_nodes):
            parent = list(self.G.predecessors(variable))
            cpd = self._simulate_single_cpd(variable, parent)
            all_cpds.append(cpd)
        return all_cpds

    def _simulate_single_cpd(self, variable, parent):
        """
        Params:
        - variable index (int)
        - parent index (int)
        """

        variable_card = self.cards[variable]
        if len(parent) == 0: 
            probs_matrix = SyntheticDataset.generate_probs_matrix(variable_card, 1)
            cpd = TabularCPD(variable=variable, variable_card=variable_card, values=probs_matrix)
        else:
            evidence_card = [self.cards[p] for p in parent]     
            m = variable_card 
            n = np.prod(evidence_card)
            probs_matrix = SyntheticDataset.generate_probs_matrix(m, n)
            
            cpd = TabularCPD(variable=variable, variable_card=variable_card, 
                        values=probs_matrix, 
                        evidence=parent, evidence_card=evidence_card)
            
        
        return cpd


    @staticmethod
    def generate_probs_matrix(m, n):
        """
        Generates a random m x n matrix where the sum of each column is 1.

        Parameters:
            m (int): The number of rows.
            n (int): The number of columns.

        Returns:
            np.ndarray: An m x n column-stochastic matrix.
        """
        # Start with a random matrix
        matrix = np.random.rand(m, n)
        # matrix = np.random.uniform(low=0.1, high=0.5, size=(m,n))
        
        # Normalize columns to sum to 1
        matrix /= matrix.sum(axis=0, keepdims=True)

        return matrix

    @staticmethod
    def simulate_er_dag(d, degree):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            # return nx.to_numpy_matrix(G)
            return nx.to_numpy_array(G)

        p = float(degree) / (d - 1)
        # Probability for edge creation
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):
        """Simulate SF DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    @staticmethod
    def simulate_random_dag(d, degree, graph_type):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = SyntheticDataset.simulate_er_dag(d, degree)
        elif graph_type == 'SF':
            B_bin = SyntheticDataset.simulate_sf_dag(d, degree)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)