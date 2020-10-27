import torch.nn as nn

class SmallSetTransformer(nn.Module):
    def __init__(self, n_outputs, n_inputs):
        super().__init__()
        num_heads = n_inputs * 4
        self.enc = nn.Sequential(
            SAB(dim_in=2, dim_out=64, num_heads=num_heads),
            SAB(dim_in=64, dim_out=64, num_heads=num_heads),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=num_heads, num_seeds=1),
            nn.Linear(in_features=64, out_features=n_outputs),
        )

    def forward(self, x):
        if x.shape[1] > 1:
            raise NotImplemented("Can't handle multiple inputs")
            batch, n_dists, n_samples, n_feats = x.shape
            x = x.reshape(batch, n_dists * n_samples, n_feats)
        else:
            x = x.squeeze(1)
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)