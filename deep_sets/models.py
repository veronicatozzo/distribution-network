import torch
import torch.nn as nn


class SmallDeepSet(nn.Module):
    def __init__(self, n_outputs=1, n_enc_layers=4, n_hidden_units=64, n_dec_layers=1, **kwargs):
        super().__init__()
        enc_layers = []
        enc_layers.append(nn.Linear(in_features=2, out_features=n_hidden_units))
        for i in range(n_enc_layers - 1):
            enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            # don't add relu to last enc layer
            if i < n_enc_layers - 2:
                enc_layers.append(nn.ReLU())
        self.enc = nn.Sequential(*enc_layers)
        dec_layers = []
        for i in range(n_dec_layers - 1):
            dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        if x.shape[1] > 1:
            raise NotImplemented("Can't handle multiple inputs")
        else:
            x = x.squeeze(1)
        return x

class SmallDeepSetMax(SmallDeepSet):
    def forward(self, x):
        x = super().forward(x)
        x = self.enc(x)
        x = torch.max(x, -2).values
        x = self.dec(x)
        return x

class SmallDeepSetMean(SmallDeepSet):
    def forward(self, x):
        x = super().forward(x)
        x = self.enc(x)
        x = x.mean(dim=-2)
        x = self.dec(x)
        return x

class SmallDeepSetSum(SmallDeepSet):
    def forward(self, x):
        x = super().forward(x)
        x = self.enc(x)
        x = x.sum(dim=-2)
        x = self.dec(x)
        return x

class TrivialMean(nn.Module):
    def __init__(self, n_outputs=1, **kwargs):
        super().__init__()
        self.dec = nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.mean(dim=-2)
        return self.dec(x)
