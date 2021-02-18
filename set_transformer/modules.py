# The code has been copied from the original 
# code in the repository https://github.com/juho-lee/set_transformer
# and it is protected under MIT Licence. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
#         if ln:
#             self.ln0 = nn.LayerNorm(dim_V)
#             self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        dim_split = 2**int(round(np.log2(dim_split),0))
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class MB(nn.Module):
    """ Moment block (MB) 

    No interactions between distributions
    
    """
    def __init__(self, n_feats, hidden_size, first_layer=False, ln=False, bn=False):
        super(MB, self).__init__()
        self.fc_q = nn.Linear(n_feats, hidden_size)
        if first_layer:
            self.fc_k = nn.Linear(n_feats, hidden_size)
        else:
            self.fc_k = nn.Linear(hidden_size, hidden_size)
        if ln:
            self.ln0 = nn.LayerNorm(hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.norm = 'ln'
        elif bn:
            # TODO: don't hard-code
            n_dists = 5
            self.ln0 = nn.BatchNorm1d(hidden_size * n_dists)
            self.ln1 = nn.BatchNorm1d(hidden_size * n_dists)
            self.norm = 'bn'
        else:
            self.norm = 'none'
#         if ln and bn:
#             raise ValueError("Don't wan't layer norm and batch norm together")
        self.fc_o = nn.Linear(hidden_size, hidden_size)


    def forward(self, Q, K):
        """ 
        Q and K are [batch, n_dist, n_samples, n_feats] 
        Q is original input (keep passing it through)
        K is previous layer output

        """
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        # element-wise multiplication
        O = Q * K
        if self.norm == 'ln':
            O = self.ln0(O)
            O = O + F.relu(self.fc_o(O))
            O = self.ln1(O)
        elif self.norm == 'bn':
            batch, n_dists, n_samples, n_feats = O.shape
            O = torch.transpose(O, 1, 2)
            O = self.ln0(O.reshape(batch * n_samples, n_dists * n_feats))
            O = O.reshape(batch, n_samples, n_dists, n_feats)
            O = O + F.relu(self.fc_o(O))
            O = self.ln1(O.reshape(batch * n_samples, n_dists * n_feats))
            O = torch.transpose(O.reshape(batch, n_samples, n_dists, n_feats), 1, 2)
        return O
    
    
class MB_norm(nn.Module):
    """ Moment block (MB) 

    No interactions between distributions
    
    """
    def __init__(self, n_feats, hidden_size, first_layer=False, ln=False, bn=False):
        super(MB_norm, self).__init__()
        self.fc_q = nn.Linear(n_feats, hidden_size)
        if first_layer:
            self.fc_k = nn.Linear(n_feats, hidden_size)
        else:
            self.fc_k = nn.Linear(hidden_size, hidden_size)
        if ln:
            self.ln0 = nn.LayerNorm(hidden_size)
            self.ln1 = nn.LayerNorm(hidden_size)
            self.lnnorm = 'ln'
        if bn:
            # TODO: don't hard-code
            n_dists = 5
            self.bn0 = nn.BatchNorm1d(hidden_size * n_dists)
            self.bn1 = nn.BatchNorm1d(hidden_size * n_dists)
            self.bnnorm = 'bn'
        else:
            self.lnnorm = 'none'
            self.bnnorm = 'none'
#         if ln and bn:
#             raise ValueError("Don't wan't layer norm and batch norm together")
        self.fc_o = nn.Linear(hidden_size, hidden_size)


    def forward(self, Q, K):
        """ 
        Q and K are [batch, n_dist, n_samples, n_feats] 
        Q is original input (keep passing it through)
        K is previous layer output

        """
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        # element-wise multiplication
        O = Q * K
        if self.lnnorm == 'ln':
            O = self.ln0(O)
            O = O + F.relu(self.fc_o(O))
            O = self.ln1(O)
        if self.bnnorm == 'bn':
            batch, n_dists, n_samples, n_feats = O.shape
            O = torch.transpose(O, 1, 2)
            O = self.bn0(O.reshape(batch * n_samples, n_dists * n_feats))
            O = O.reshape(batch, n_samples, n_dists, n_feats)
            O = O + F.relu(self.fc_o(O))
            O = self.bn1(O.reshape(batch * n_samples, n_dists * n_feats))
            O = torch.transpose(O.reshape(batch, n_samples, n_dists, n_feats), 1, 2)
        return O