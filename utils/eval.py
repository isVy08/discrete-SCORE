import os
import random
import numpy as np
import torch
import copy
import pandas as pd
import networkx as nx
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.SHD import SHD

def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    # tf.compat.v1.set_random_seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass

def print_edges(B, k = 10):
    edges = np.where(B)
    N = edges[0].shape[0]
    k = min(k, N)
    for i in range(k):
        print(f"({edges[0][i]},{edges[1][i]})", end=' ')
    print()



def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(
        weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres


def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    B_bin = (abs(B) >= graph_thres).astype(np.float32)

    return B, B_bin

def to_dag(B, graph_thres):

    def prune(B, graph_thres):
        B = np.copy(B)
        B[np.abs(B) < graph_thres] = 0    # Thresholding

        B_bin = (abs(B) >= graph_thres).astype(np.float32)

        return B_bin
    
    if graph_thres is None:
        graph_thres = 0.01
        completed = False
        while not completed:
            B_processed_bin = prune(B, graph_thres)
            graph_thres += 0.01
            if is_dag(B_processed_bin):
                completed = True
                print('Optimal threshold:', graph_thres)
    else: 
        B_processed_bin = prune(B, graph_thres)

    return B_processed_bin

def adjmat_to_cpdag_obj(adj_mat, is_dag=True):
    '''
    adjmat: adjacency matrix for dag or cpdag
    '''
    nodes = [GraphNode('X' + str(i + 1)) for i in range(adj_mat.shape[0])]
    cpdag = GeneralGraph(nodes)
    if is_dag:
        G = np.zeros_like(adj_mat)
        for i in range(adj_mat.shape[0]):
            for j in range(adj_mat.shape[0]):
                if adj_mat[i, j] == 1 and adj_mat[j, i] == 0:
                    G[i,j] = -1
                    G[j,i] = 1
                elif adj_mat[j, i] == 1 and adj_mat[i, j] == 0:
                    G[j,i] = -1
                    G[i,j] = 1
                elif adj_mat[i, j] == adj_mat[j, i] == 1: 
                    G[j,i] = G[i,j] = -1
        adj_mat = G
    
    cpdag.graph = adj_mat
    return cpdag

def cpdag_to_dag(cpdag):
    dag = np.zeros_like(cpdag)
    for i in range(dag.shape[0] - 1):
        for j in range(i, dag.shape[0]):
            if cpdag[j,i] == 1 and cpdag[i,j] == -1:    # i -> j
                dag[i,j] = 1
            elif cpdag[i,j] == 1 and cpdag[j,i] == -1:  # i <- j
                dag[j,i] = 1
    return dag

class MetricsCPDAG(object):
    def __init__(self, B_est, B_true):

        B_true = adjmat_to_cpdag_obj(B_true)
        shd = SHD(B_true, B_est).get_shd()
        adj = AdjacencyConfusion(B_true, B_est)
        adjPrec = adj.get_adj_precision()
        adjRec = adj.get_adj_recall()
        
        
        adjF1 = 2*(adjRec*adjPrec)/(adjRec+adjPrec)

        arrow = ArrowConfusion(B_true, B_est)
        arrowPrec = arrow.get_arrows_precision()
        arrowRec = arrow.get_arrows_recall()
        arrowF1 = 2*(arrowRec*arrowPrec)/(arrowRec+arrowPrec)

        self.metrics = {'adjPrec':adjPrec, 'adjRec':adjRec, 'adjF1':adjF1,  
                        'arrowPrec': arrowPrec, 'arrowRec':arrowRec, 'arrowF1':arrowF1, 'shd': shd}
    
    def display(self):
        for metric, value in self.metrics.items():
            print(f'{metric} : {value:.5g}')



class MetricsDAG(object):
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.
    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true):
        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)
    
    def display(self):
        for k, v in self.metrics.items():
            print(k, ':', v)

    @staticmethod
    def _count_accuracy(B_est, B_true, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        decimal_num: int
            Result decimal numbers.
        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0

        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]

        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(
            pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(
            pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(
            cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {
            'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'nnz': pred_size,
              'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore, 'shd': shd}
        for i in mt:
            mt[i] = round(mt[i], decimal_num)

        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        num_true = W_true.sum(axis=1).sum()
        assert num_true != 0

        # true_positives
        num_tp = (W_p + W_true).applymap(lambda elem: 1 if elem ==
                                         2 else 0).sum(axis=1).sum()
        # False Positives + Reversed Edges
        num_fn_r = (W_p - W_true).applymap(lambda elem: 1 if elem ==
                                           1 else 0).sum(axis=1).sum()
        score = np.max((num_tp-num_fn_r, 0))/num_true

        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert(W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
        TP = (W_p + W_true).applymap(lambda elem: 1 if elem ==
                                     2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/TP_FP
        recall = TP/TP_FN
        F1 = 2*(recall*precision)/(recall+precision)

        return precision, recall, F1

def evaluate(B_est, B_true, weighted=False, cpdag = False):
    
    if cpdag:
        raw_result = MetricsCPDAG(B_est, B_true)
    else:
        if weighted:
            _, B_est = postprocess(B_est, graph_thres = 0.3)
        print(B_est.max(), B_est.min())
        print('Is DAG?', is_dag(B_est))
        raw_result = MetricsDAG(B_est, B_true)
    raw_result.display()
    return raw_result


def write_result(result_dict, config_code, saved_path):
    print('Writing results ...')
    file = open(saved_path, 'a+')
    file.write(f'{config_code}\n')

    for k, v in result_dict.metrics.items():
        file.write(f'{k} : {v}\n')


    file.write('======================\n')
    file.close()
