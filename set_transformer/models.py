import torch
import torch.nn as nn
import torch

from set_transformer.modules import SAB, PMA, MB

class SmallSetTransformer(nn.Module):
    def __init__(self, n_outputs=1, n_inputs=1, n_enc_layers=2, n_hidden_units=64, n_dec_layers=2, **kwargs):
        super().__init__()
        num_heads = n_inputs * 4
        enc_layers = []
        enc_layers.append(SAB(dim_in=2, dim_out=n_hidden_units, num_heads=num_heads))
        for i in range(n_enc_layers - 1):
            enc_layers.append(SAB(dim_in=n_hidden_units, dim_out=n_hidden_units, num_heads=num_heads))
        self.enc = nn.Sequential(*enc_layers)
        dec_layers = []
        for j in range(n_dec_layers - 1):
            dec_layers.append(PMA(dim=n_hidden_units, num_heads=num_heads, num_seeds=1))
        dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        if x.shape[1] > 1:
            encoded = []
            for j in range(x.shape[1]):
                a = x[:, j, :, :].squeeze(1)
                encoded.append(self.enc(a))
            x = torch.cat(encoded, 1)
        else:
            x = x.squeeze(1)
            x = self.enc(x)
        x = self.dec(x)
        # TODO: change squeeze(1) for multiple inputs
        return x.squeeze(-1).squeeze(1)


class SmallDeepSamples(nn.Module):
    def __init__(self, n_outputs=1, n_inputs=1, n_enc_layers=2, n_hidden_units=64, n_dec_layers=2, **kwargs):
        super().__init__()
        self.enc_layers = []
        self.enc_layers.append(MB(n_feats=2, hidden_size=n_hidden_units, first_layer=True))
        for i in range(n_enc_layers - 1):
            self.enc_layers.append(MB(n_feats=2, hidden_size=n_hidden_units))
        self.enc_layers = nn.ModuleList(self.enc_layers)
        self.dec = nn.Linear(in_features=n_hidden_units * n_inputs, out_features=n_outputs)

    def forward(self, x, lengths):
        """ lengths: [batch, n_dists] """
        batch, n_dists, n_samples, n_feats = x.shape
        out = x
        for layer in self.enc_layers:
            out = layer(x, out)
        # [batch, n_dists, n_samples]
        length_mask = torch.arange(n_samples).expand(lengths.shape[0], lengths.shape[1], n_samples).to("cuda:0") < lengths.unsqueeze(-1)
        out = (out * length_mask.unsqueeze(-1)).sum(dim=-2) / length_mask.sum(dim=-1).unsqueeze(-1)
        return self.dec(out.reshape(batch, -1))
