import torch.nn as nn

from set_transformer.modules import SAB, PMA

class SmallSetTransformer(nn.Module):
    def __init__(self, n_outputs=1, n_inputs=1, **kwargs):
        super().__init__()
        # Note: SetTransformer uses four-head attention per distribution
        self.enc = nn.Sequential(
            SAB(dim_in=2, dim_out=64, num_heads=4 * n_inputs),
            SAB(dim_in=64, dim_out=64, num_heads=4 * n_inputs),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4 * n_inputs, num_seeds=1),
            nn.Linear(in_features=64, out_features=n_outputs),
        )

    def forward(self, x):
        batch, num_dists, num_samples, n_feats = x.shape
        x = x.reshape(batch, num_dists * num_samples, n_feats)
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)
