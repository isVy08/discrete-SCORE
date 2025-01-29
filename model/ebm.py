import torch
import torch.nn as nn
import torch.nn.functional as F

def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    Get time embedding for timesteps in PyTorch.
    """
    assert embedding_dim % 2 == 0
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(max_positions, dtype=torch.float32, device=timesteps.device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]  # Expand dims to align for broadcasting
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class BinaryMLPScoreFunc(nn.Module):
    """Get a scalar score for an input."""
    def __init__(self, num_layers, hidden_size, time_scale_factor=1000.0):
        super(BinaryMLPScoreFunc, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_scale_factor = time_scale_factor
        
        # Define MLP layers
        self.dense_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, t):
        temb = transformer_timestep_embedding(t * self.time_scale_factor, self.hidden_size)
        x = x.float()
        
        for layer in self.dense_layers:
            x = layer(x) + temb
            x = F.elu(x)
        
        x = self.output_layer(x)
        return x


class BinaryTransformerScoreFunc(nn.Module):
    """Get a scalar score for an input."""
    def __init__(self, config):
        super(BinaryTransformerScoreFunc, self).__init__()
        self.config = config
        self.transformer = MaskedTransformer(config)  # Assuming MaskedTransformer is implemented in PyTorch
    
    def forward(self, x, t):
        temb = transformer_timestep_embedding(t * self.config.time_scale_factor, self.config.embed_dim)
        x = x.int()
        cls_token = torch.ones((x.shape[0], 1), dtype=torch.int32) * self.config.vocab_size
        x = torch.cat([cls_token, x], dim=1)
        score = self.transformer(x, temb, 0)[..., 0]
        return score


class CatMLPScoreFunc(nn.Module):
    """Get a scalar score for an input."""
    def __init__(self, vocab_size, seq_length, cat_embed_size, num_layers, hidden_size, time_scale_factor=1000.0):
        super(CatMLPScoreFunc, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.cat_embed_size = cat_embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_scale_factor = time_scale_factor
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, cat_embed_size)
        
        # Define MLP layers
        self.input_layer = nn.Linear(seq_length * cat_embed_size, hidden_size)
        self.dense_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, t):
        
        
        temb = transformer_timestep_embedding(t * self.time_scale_factor, self.hidden_size)
        e = self.embedding(x)
        x = e.view(x.shape[0], -1)
        x = self.input_layer(x)
        
        for layer in self.dense_layers:
            x = layer(x) + temb
            x = F.silu(x)
        
        x = self.output_layer(x)
        return e, x
        

class BinaryScoreModel(nn.Module):
    """EBM for binary data."""

    def __init__(self, config):
        super(BinaryScoreModel, self).__init__()
        if config.net_arch == 'mlp':
            self.net = BinaryMLPScoreFunc(
                num_layers=config.num_layers, hidden_size=config.embed_dim,
                time_scale_factor=config.time_scale_factor)
        elif config.net_arch == 'transformer':
            self.net = BinaryTransformerScoreFunc(config)
        else:
            raise ValueError('Unknown net arch: %s' % config.net_arch)
        self.fwd_model = get_fwd_model(config)  # Assuming get_fwd_model is a function that returns a PyTorch model

    def get_q(self, params, xt, t):
        bsize = xt.shape[0]
        ddim = self.config.discrete_dim
        qxt = self.net(params, xt, t)
        mask = torch.eye(ddim).repeat(bsize, 1)
        xrep = xt.repeat(ddim, 1)
        xneg = (mask - xrep) * mask + (1 - mask) * xrep
        t = t.repeat(ddim)
        qxneg = self.net(params, xneg, t)
        qxt = qxt.repeat(ddim, 1)
        return qxneg, qxt

    def get_logits(self, params, xt, t):
        bsize = xt.shape[0]
        qxneg, qxt = self.get_q(params, xt, t)
        qxneg = qxneg.view(-1, bsize).T
        qxt = qxt.view(-1, bsize).T
        xt_onehot = F.one_hot(xt, num_classes=2)
        qxneg, qxt = qxneg.unsqueeze(-1), qxt.unsqueeze(-1)
        logits = xt_onehot * qxt + (1 - xt_onehot) * qxneg
        return logits

    def get_ratio(self, params, xt, t, xt_target=None):
        qxneg, qxt = self.get_q(params, xt, t)
        ratio = torch.exp(qxneg - qxt)
        return ratio.view(-1, xt.shape[0]).T

    def get_logprob(self, params, xt, t, xt_target=None):
        logits = self.get_logits(params, xt, t)
        return F.log_softmax(logits, dim=-1)

    def loss(self, params, rng, x0, xt, t):
        _, ll_xt = self.get_logprob(params, xt, t)
        loss = -torch.sum(ll_xt) / xt.shape[0]
        aux = {'loss': loss}
        return loss, aux


class CategoricalScoreModel(nn.Module):
    """EBM for categorical data."""

    def __init__(self, config):
        super(CategoricalScoreModel, self).__init__()
        self.config = config
        if config.net_arch == 'mlp':
            if config.vocab_size == 2:
                self.net = BinaryMLPScoreFunc(
                    num_layers=config.num_layers, hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor)
            else:
                self.net = CatMLPScoreFunc(
                    vocab_size=config.vocab_size, seq_length=config.discrete_dim, 
                    cat_embed_size=config.cat_embed_size,
                    num_layers=config.num_layers, hidden_size=config.embed_dim,
                    time_scale_factor=config.time_scale_factor)
        else:
            raise ValueError('Unknown net arch: %s' % config.net_arch)

    def forward(self, xt, t):
        bsize = xt.shape[0]
        ddim = self.config.discrete_dim
        vocab_size = self.config.vocab_size
        mask = torch.eye(ddim).repeat_interleave(bsize * vocab_size, dim=0).to(xt.device)
        xrep = xt.repeat(ddim * vocab_size, 1)
        candidate = torch.arange(vocab_size).repeat_interleave(bsize, dim=0).to(xt.device)
        candidate = candidate.repeat(ddim).unsqueeze(1)
        xall = mask * candidate + (1 - mask) * xrep
        t = t.repeat(ddim * vocab_size)
        embs, qall = self.net(xall.long(), t)

        logits = qall.view(ddim, vocab_size, bsize).transpose(0, 2).transpose(1, 2)
        ll_all = F.log_softmax(logits, dim=-1)
        ll_xt = ll_all[torch.arange(bsize)[:, None], torch.arange(self.config.discrete_dim)[None, :], xt]
        return ll_all, ll_xt
        
    def loss(self,xt, t):
        _, ll_xt = self.forward(xt, t)
        loss = -torch.sum(ll_xt) / xt.shape[0]
        return loss

if __name__ == "__main__":
    import argparse

    # Define configuration for the test model
    config = argparse.Namespace(
        net_arch='mlp',
        vocab_size=10,
        cat_embed_size=64,
        num_layers=2,
        embed_dim=128,
        time_scale_factor=1000.0,
        discrete_dim=5
    )

    # Generate test data
    batch_size = 8
    sequence_length = config.discrete_dim
    vocab_size = config.vocab_size

    # xt: Categorical data (random integers in the range [0, vocab_size-1])
    xt = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)

    # t: Timesteps (random integers to simulate time steps)
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.long)

    # Create the model
    model = CategoricalScoreModel(config)

    # Test forward pass
    score = model.net(xt, t)

    # Test loss 
    loss = model.loss(xt, xt)

    print('Shape of score:', score.shape)
    print('Shape of loss:', loss.shape)

