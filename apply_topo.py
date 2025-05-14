import numpy as np
import pandas as pd
from utils.topo import *
from utils.eval import cpdag_to_dag, is_dag


def estimate_dag(dataset, method, ordered_vertices, path_to_dag=None):

    from causalai.data.tabular import TabularData
    from causalai.models.tabular.pc import PC
    from causalai.models.common.prior_knowledge import PriorKnowledge
    from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests

    data = dataset.X.values

    def estimate_dag(result):
        B_est = np.zeros_like(dataset.B_bin)
        for key in result.keys():
            parents = result[key]['parents']
            for pa in parents:
                B_est[int(key), int(pa)] = 1
        return B_est

    if method == 'PC':
        
        var_names = [str(i) for i in range(dataset.num_nodes) ]
        data_obj = TabularData(data, var_names=var_names)

        forbidden_links = generate_forbidden_links(ordered_vertices[::-1], k = 1)
        prior_knowledge = PriorKnowledge(forbidden_links=forbidden_links)
        CI_test = DiscreteCI_tests(method="log-likelihood")  
        pc = PC(data=data_obj, prior_knowledge=prior_knowledge, CI_test=CI_test,use_multiprocessing=True)
        result = pc.run(pvalue_thres=0.01, max_condition_set_size=2)
        B_est = estimate_dag(result)

    elif 'GES' in method: 
        from causallearn.score.LocalScoreFunction import local_score_BDeu
        from causallearn.search.ScoreBased.GES import ges
        
        result = ges(data, "local_score_BDeu")
        cpdag_obj = result['G']
        B_est = cpdag_to_dag(cpdag_obj.graph)
        
        func = local_score_BDeu
        X = dataset.X.values 
        B_new = B_est.copy()
        for i in range(B_est.shape[0]):
            curr_pa = np.argwhere(B_est[:, i])[:, 0].tolist() 
            curr_scr = func(X, i, curr_pa)
            candidates = np.argwhere(1 - B_est[:, i])[:, 0].tolist() 
            for j in candidates: 
                if B_est[i,j] == 0 and (ordered_vertices.index(j) < ordered_vertices.index(i)):
                    newpa = curr_pa + [j] # include this edge if there is an update in score, but do not alter curre
                    scr = func(X, i, newpa)
                    delta = curr_scr - scr
                    curr_B = B_new.copy()
                    curr_B[j,i] = 1.
                    dagness = is_dag(curr_B)
                    if delta > 0 and dagness:
                        B_new[j,i] = 1.
                        curr_scr = scr
                        curr_pa = newpa
        B_est = B_new    
    else:

        assert path_to_dag is not None, 'Pre-estimated adjacency matrix is required!'
        B_est = pd.read_csv(path_to_dag, index_col=False)
        A = full_DAG(ordered_vertices)
        B_est = B_est * A
                
    return B_est    
        
