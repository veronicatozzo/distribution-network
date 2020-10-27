import torch.nn as nn

class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=2, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        if x.shape[1] > 1:
            raise NotImplemented("Can't handle multiple inputs")
        else:
            x = x.squeeze(1)
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)