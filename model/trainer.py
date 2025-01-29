import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def find_leaf(model, xt, t):
    logprob, _ = model(xt, t)

    scores = []
    num_nodes = xt.shape[1]
    for i in range(num_nodes):
        logits = logprob[:, i, :]
        score = torch.sum( (logits ** 2) * logits.exp(), dim=1) - torch.sum(logits * logits.exp(), dim=1) ** 2
        values = torch.unique(xt[:, i], return_counts=False)
        score_per_node = torch.sum(score) / len(values)
        scores.append(score_per_node.item())
    leaf = np.array(scores).argmin()
    return leaf

class Trainer:
    
    def __init__(self, dataset, batch_size):
        
        
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        dataset.X = dataset.X.loc[:, list(range(dataset.num_nodes))]
        data = dataset.X.values.copy()

        self.data = torch.LongTensor(data)

        self.indices = list(range(dataset.num_samples))
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
        t = t.to(xt.device)

        loader = DataLoader(self.indices, batch_size=self.batch_size, shuffle=True)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            total_loss = 0
            for ids in loader:
                xt_batch = xt[ids, :]
                t_batch = t[ids,]
                
                loss = model.loss(xt_batch, t_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            total_loss /= len(loader)
    
        pred_leaf = find_leaf(model, xt, t)
        correct = True if pred_leaf in true_leaf else False   
        return pred_leaf, correct