import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def compute_score_sum(logits, method):
    if method == 'entropy':
        # v1: entropy
        score = torch.sum(logits * logits.exp(), dim=1) 
    elif method == 'variance':
        # v2: variance 
        score = torch.sum( (logits ** 2) * logits.exp(), dim=1) - torch.sum(logits * logits.exp(), dim=1) ** 2
    elif method == 'renyi2': 
        # v3: renyi entropy with alpha = 2   
        score = torch.sum(logits ** 2, dim=1)                 
    elif method == 'renyi0':
        # v4: KL(u|p): renyi entropy with alpha = 0
        score = - torch.sum(logits, dim=1)                    
    else:
        # v5: KL(p|u) + KL(u|p)
        score = - torch.sum(logits, dim=1) + torch.sum(logits * logits.exp(), dim=1) 
    score = torch.sum(score)
    return score
            

def find_leaf(model, xt, t, loader, device, method='entropy'):
    
    scores = []
    num_nodes = xt.shape[1]
    score_per_node = [0] * num_nodes
    for ids in loader: 
        xt_batch = xt[ids, :].to(device)
        t_batch = t[ids,].to(device)
        logprob, _ = model(xt_batch, t_batch)
        for i in range(num_nodes):    
            logits = logprob[:, i, :]
            score = compute_score_sum(logits, method)
            values = torch.unique(xt[:, i], return_counts=False)
            score_per_node[i] += score.item() / len(values)

    scores = score_per_node
    leaf = np.array(scores).argmin()
    return leaf


class Trainer:

    def __init__(self, dataset, batch_size, sample_size=None):
        
        
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = 'entropy'
        
        if sample_size is not None:
            dataset.X = dataset.X.sample(frac=1).reset_index(drop=True)
            dataset.X = dataset.X.loc[:sample_size, list(range(dataset.num_nodes))]

        else: 
            dataset.X = dataset.X.loc[:, list(range(dataset.num_nodes))]
        
        data = dataset.X.values.copy()


        
        self.data = torch.LongTensor(data)

        self.indices = list(range(dataset.X.shape[0]))
        self.batch_size = batch_size
        

    def _prepare_data(self, curr_order=[]):
        keep_indices = [i for i in range(self.dataset.num_nodes) if i not in curr_order]
        xt  = self.data[:, keep_indices]
        b_bin = self.dataset.B_bin[keep_indices][:, keep_indices]
        true_leaf = np.argwhere(b_bin.sum(1)==0).squeeze().tolist()

        if isinstance(true_leaf, int):
            true_leaf = [true_leaf]

        return xt, true_leaf, keep_indices


    def train(self, xt, model, optimizer, true_leaf, num_timesteps, num_epochs):

        t = torch.randint(0, num_timesteps, (xt.shape[0],), dtype=torch.long)

        loader = DataLoader(self.indices, batch_size=self.batch_size, shuffle=False)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            total_loss = 0
            for ids in loader:
                xt_batch = xt[ids, :].to(self.device)
                t_batch = t[ids,].to(self.device)
            
                loss = model.loss(xt_batch, t_batch) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            total_loss /= len(loader)


        pred_leaf = find_leaf(model, xt, t, loader, self.device, self.method)
        correct = True if pred_leaf in true_leaf else False   
        return pred_leaf, correct
