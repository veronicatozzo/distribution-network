import torch.nn as nn


class SmallDeepSet(nn.Module):
    def __init__(self, n_outputs=1, n_inputs=1, **kwargs):
        #change to take more than one input into account
        
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_outputs),
        )

    def forward(self, x):
        print(x.shape)
        if x.shape[1] > 1:
            encoded = []
            for j in range(x.shape[1]):
                a = x[:, j, :, :].squeeze(1)
                print(a.shape)
                encoded.append(self.enc(a))
            x = torch.cat(encoded, 1)
            print(x.shape)
        else:
            x = x.squeeze(1)
            x = self.enc(x)
        return x

class SmallDeepSetMax(SmallDeepSet):
    def forward(self, x):
        x = super().forward(x)
        #x = self.enc(x)
        x = x.max(dim=-2)
        x = self.dec(x)
        return x

class SmallDeepSetMean(SmallDeepSet):
    def forward(self, x):
        x = super().forward(x)
        #x = self.enc(x)
        x = x.mean(dim=-2)
        x = self.dec(x)
        return x

class SmallDeepSetSum(SmallDeepSet):
    def forward(self, x):
        x = super().forward(x)
        #x = self.enc(x)
        x = x.sum(dim=-2)
        x = self.dec(x)
        return x

