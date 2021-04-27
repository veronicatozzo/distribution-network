import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


class EncodingBlock(nn.Module):
    def __init__(self, n_inputs=2, n_hidden_units_global=8,         
                 n_hidden_units_sample=256, n_layers_global=3, n_layers_sample=3, 
                 activation=nn.ReLU, **kwargs):
        super().__init__(**kwargs)
        
        enc_layers_global = []
        for i in range(n_layers_global):
            if i == 0:
                enc_layers_global.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units_global))
                enc_layers_global.append(activation())
            else:
                enc_layers_global.append(nn.Linear(in_features=n_hidden_units_global, out_features=n_hidden_units_global))
                enc_layers_global.append(activation())
        enc_layers_global= enc_layers_global[:-1]
        self.enc_global = nn.Sequential(*enc_layers_global)
        
        enc_layers_sample = []
        for i in range(n_layers_sample):
            if i == 0:
                enc_layers_sample.append(nn.Linear(in_features=n_hidden_units_global+n_inputs, out_features=n_hidden_units_sample))
                enc_layers_sample.append(activation())
            else:
                enc_layers_sample.append(nn.Linear(in_features=n_hidden_units_sample, out_features=n_hidden_units_sample))
                enc_layers_sample.append(activation())
        # remove last relu
        enc_layers_sample = enc_layers_sample[:-1]
        self.enc_sample = nn.Sequential(*enc_layers_sample)

    def forward(self, x, length=None):
        global_features = self.enc_global(x)
        global_features = global_features.mean(dim=-2)
        x = torch.cat([x, torch.repeat_interleave(global_features[:, np.newaxis, :], x.shape[1], 1)], axis=2)
        x = self.enc_sample(x)
        x = x.mean(dim=-2)
        self.global_features = global_features
        return x
    

class DeepSamples(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_rep_encoding_block=1,     
                 n_enc_layers_outer=4, n_enc_layers_inner=4,   n_dec_layers=1, n_hidden_units_outer=64, n_hidden_units_inner=64, normalization=True,connect_decoder=True,  activation=nn.ReLU,  n_samples=1000,
                 n_dists=1, **kwargs):
        super().__init__(**kwargs)
        
        
        self.encoders = []
        for i in range(n_dists):
            aux = []
            for r in range(n_rep_encoding_block):
                aux.append(EncodingBlock(n_inputs=n_inputs, n_hidden_units_global=n_hidden_units_outer,         
                 n_hidden_units_sample=n_hidden_units_inner, n_layers_global=n_enc_layers_outer, n_layers_sample=n_enc_layers_inner, 
                 activation=activation))
            self.encoders.append(nn.Sequential(*aux))

        dec_layers = []
        for i in range(n_dec_layers):
            if i == 0:
                if connect_decoder:
                    if normalization:
                        dec_layers.append(nn.Linear(in_features=(n_hidden_units_inner+ n_hidden_units_outer + n_inputs)*n_dists, out_features=n_hidden_units_inner))
                    else:
                        dec_layers.append(nn.Linear(in_features=(n_hidden_units_inner + n_hidden_units_outer)*n_dists, out_features=n_hidden_units_inner))
                else:
                    if normalization:
                        dec_layers.append(nn.Linear(in_features=(n_hidden_units_inner+ n_inputs)*n_dists, out_features=n_hidden_units_inner))
                    else:
                        dec_layers.append(nn.Linear(in_features=n_hidden_units_inner*n_dists, out_features=n_hidden_units_inner))
                dec_layers.append(activation())
            if i == n_dec_layers- 1:
                dec_layers.append(nn.Linear(in_features=n_hidden_units_inner, out_features=n_outputs))
            else:
                dec_layers.append(nn.Linear(in_features=n_hidden_units_inner, out_features=n_hidden_units_inner))
                dec_layers.append(activation())
        self.dec = nn.Sequential(*dec_layers)
        self.normalization=normalization
        self.connect_decoder=connect_decoder
        self.n_dists=n_dists

    def forward(self, x, length=None):
        
        if len(x.shape) == 4 and x.shape[1] > 1:

            assert x.shape[1] == self.n_dists
            encoded = []
            means = []
            for j in range(x.shape[1]):
                a = x[:, j, :, :].squeeze(1)
                if self.normalization:
                    means.append(torch.mean(a, axis=1))
                    a -= means[-1].unsqueeze(1)
                encoded.append(self.encoders[j](a))
            x = torch.cat(encoded, 1)

            if self.connect_decoder:
                global_features = []
                for j in range(x.shape[1]):
                    global_features.append(self.encoders[j].global_features)
                feats = torch.cat(global_features, 1)
                x = torch.cat([x, feats], axis=1)  # [b, hidden_outer*n_dists + hidden_inner*n_dists]
            if self.normalization:
                means = torch.cat(means, 1)
                x = torch.cat([x, means], axis=1) 
        else:
            assert self.n_dists == 1
            x = x.squeeze(1)
            if self.normalization:
                means = torch.mean(a, axis=1)
                x -= means.unsqueeze(1)
            x = self.encoders[0](x)
            if self.connect_decoder:
                x = torch.cat([x, self.encoders[0].global_features], axis=1)  # [b, hidden + features_per_sample]
            if self.normalization:
                x = torch.cat([x, means], axis=1) 
        x = self.dec(x)
        return x
       
    
    