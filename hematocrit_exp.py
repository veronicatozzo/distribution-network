from scipy.stats import kurtosis, skew
from torch.utils.data import Dataset
from sklearn.datasets import make_spd_matrix
from sklearn.covariance import empirical_covariance
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import numpy as np
import itertools
import torch.nn.functional as F
#from .set_transformer.models import SmallSetTransformer, SmallDeepSamples
import torch.nn as nn
import torch
import copy
import matplotlib.pyplot as plt 
import os
from scipy.special import logsumexp
import math
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF

from utils import str_to_bool_arg, QuantileScaler

from deep_samples.model import DeepSamples

from src.dataset import FullLargeDataset



class BasicDeepSet(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=4, n_hidden_units=64, n_dec_layers=1, 
                 multiplication=True,ln=False, bn=False, activation=nn.ReLU, instance_norm=False, sample_norm=False, n_samples=1000, **kwargs):
        """ Note: sample_norm = True first tranposes the data so that the sample_dim is last to reuse existing norm implementations """
        super().__init__()
        if sample_norm and any([bn, ln, instance_norm]):
            raise ValueError("Cannot have sample_norm and other norms")
        enc_layers = []
       for i in range(n_enc_layers):
            if i == 0:
                if sample_norm:
                    enc_layers.append(nn.ConvTranspose1d(n_inputs, n_hidden_units, 1))
                else:
                    enc_layers.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units))
            else:
                if sample_norm:
                    enc_layers.append(nn.ConvTranspose1d(n_hidden_units, n_hidden_units, 1))
                else:
                    enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            if ln:
                enc_layers.append(nn.LayerNorm(n_hidden_units))
            if bn:
                enc_layers.append(nn.BatchNorm1d(n_samples))
            if instance_norm:
                enc_layers.append(nn.InstanceNorm1d(n_samples))
            if sample_norm:
                if i == 0:
                    enc_layers.append(nn.InstanceNorm1d(n_hidden_units, affine=True))
            enc_layers.append(activation())
        # remove last relu
        enc_layers = enc_layers[:-1]
        self.enc = nn.Sequential(*enc_layers)
        dec_layers = []
        # for i in range(n_dec_layers - 1):
        #     dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        #     dec_layers.append(activation())
        # dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for i in range(n_dec_layers):
            if i == n_dec_layers - 1:
                dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
            else:
                dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
                if ln:
                    dec_layers.append(nn.LayerNorm(n_hidden_units))
                dec_layers.append(activation())
        self.dec = nn.Sequential(*dec_layers)
        self.multiplication=multiplication
        self.sample_norm = sample_norm

    def forward(self, x):
        if len(x.shape) == 4 and x.shape[1] > 1:
            encoded = []
            for j in range(x.shape[1]):
                a = x[:, j, :, :].squeeze(1)
                if self.sample_norm:
                    encoded.append(torch.transpose(self.enc(torch.transpose(a, 1, 2)), 1, 2))
                else:
                    encoded.append(self.enc(a))
            x = torch.cat(encoded, 1)
        else:
            x = x.squeeze(1)
            if self.sample_norm:
                out = torch.transpose(self.enc(torch.transpose(x, 1, 2)), 1, 2)
            else:
                out = self.enc(x)
            #x = torch.mul(x, out)
        return out
    
class BasicDeepSetMean(BasicDeepSet):
    def forward(self, x, length=None):
        x = super().forward(x)
        if self.sample_norm:
            x = super().forward(torch.transpose(x, 1, 2))
        else:
            x = super().forward(x)
        if self.multiplication:
            x = torch.mul(x, x)
        if self.sample_norm:
            x = torch.transpose(x, 1, 2)
        x = x.mean(dim=-2)
        x = self.dec(x)
        return x

class BasicDeepSetMeanRC(BasicDeepSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dec_layers = []
        for i in range(kwargs['n_dec_layers']):
            if i == kwargs['n_dec_layers'] - 1:
                dec_layers.append(nn.Linear(in_features=kwargs['n_hidden_units'] + kwargs['n_inputs'], out_features=kwargs['n_outputs']))
            else:
                dec_layers.append(nn.Linear(in_features=kwargs['n_hidden_units'] + kwargs['n_inputs'], out_features=kwargs['n_hidden_units'] + kwargs['n_inputs']))
                if kwargs['ln']:
                    dec_layers.append(nn.LayerNorm(kwargs['n_hidden_units'] + kwargs['n_inputs']))
                dec_layers.append(kwargs['activation']())
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, length=None):
#         x = super().forward(x)
        means = torch.mean(x, axis=1)
        # print(means.shape)
        x -= means.unsqueeze(1)
        if self.sample_norm:
            x = self.enc(torch.transpose(x, 1, 2))
        else:
            x = self.enc(x)
        if self.multiplication:
            x = torch.mul(x, x)
        if self.sample_norm:
            x = torch.transpose(x, 1, 2)
        x = x.mean(dim=-2)
        x = torch.cat([x, means], axis=1)  # [b, hidden + features_per_sample]
        # print('x', x.shape)
        x = self.dec(x)
        return x

    
class BasicDeepSetSum(BasicDeepSet):
    def forward(self, x, length=None):
#         x = super().forward(x)
        if self.sample_norm:
            x = self.enc(torch.transpose(x, 1, 2))
        else:
            x = self.enc(x)
        if self.multiplication:
            x = torch.mul(x, x)
        if self.sample_norm:
            x = torch.transpose(x, 1, 2)
        x = x.sum(dim=-2)
        x = self.dec(x)
        return x

    



def train_nn(model, name, optimizer, scheduler, train_generator, test_generator, classification=False, 
             n_epochs=10, outputs=[], use_wandb=False, plot_gradients=False, seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_wandb:
        import wandb
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(classification)
    model = model.to(device)

    if use_wandb and plot_gradients:
        wandb.watch(model, log='all')
    # by default, reduction = mean when multiple outputs
    #criterion = nn.MSELoss() 
    if classification:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss(reduction="none") 
    step = 0
    best_loss_ts = None
    best_loss_tr = None
    losses_tr = []
    losses_ts = []
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for epoch in range(n_epochs):
        print(epoch)
        train_aux = []
        for x, y, lengths in train_generator:
            # print(x.shape)
            x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
            preds = model(x, lengths)
            preds = preds.reshape(x.shape[0], len(outputs))
            assert preds.shape == y.shape, "{} {}".format(preds.shape, y.shape)
            loss_elements = criterion(preds, y)
            loss = loss_elements.mean()
            if np.isnan(loss.detach().cpu().numpy()):
                raise ValueError("Train loss is nan: ", loss)
            train_aux.append(loss.detach().cpu().numpy())
            # TODO: maybe we don't want to log at every step
            if use_wandb:
                wandb.log({f"{name} train loss per step": loss}, step=step)
            if len(outputs) > 1:
                outputs_loss = loss_elements.mean(dim=0)
                # print(outputs)
                # print(outputs_loss)
                assert len(outputs) == len(outputs_loss)
                per_output_loss = outputs_loss
                if use_wandb:
                    for i in range(len(outputs)):
                        wandb.log({outputs[i]: per_output_loss[i]}, step=step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = model(x, lengths)
            step += 1
            if step % 20 == 0:
                # losses_tr.append(per_output_loss.detach().cpu().numpy())
                
                aux = []
                accuracy = []
                for x, y, lengths in test_generator:
                    x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
                    loss_elements = criterion(model(x, lengths), y)
                    loss = loss_elements.mean()
                    if np.isnan(loss.detach().cpu().numpy()):
                        raise ValueError("Test loss is nan: ", loss)
                    if classification:
                        accuracy.append(accuracy_score(model(x, lengths).detach().cpu().numpy(),
                                                       y.detach().cpu().numpy().astype(np.int8)))
                    aux.append(loss.detach().cpu().numpy())
                test_loss = np.nanmean(aux)
                if use_wandb:
                    wandb.log({f"{name} test loss per step": test_loss}, step=step)
                if len(outputs) > 1:
                    outputs_loss = loss_elements.mean(dim=0)
                    assert len(outputs) == len(outputs_loss)
                    per_output_loss = outputs_loss
                    if use_wandb:
                        for i in range(len(outputs)):
                            wandb.log({outputs[i]: per_output_loss[i]}, step=step)
                train_loss = train_aux[-1]
#                 train_loss = np.nanmean(train_aux)
#                 print(train_aux)
                train_aux = []
                losses_tr.append(train_loss)
#                 print(train_loss)
                if not np.isnan(train_loss) and not best_loss_tr or (train_loss < best_loss_tr):
                    if use_wandb:
                        wandb.run.summary["best_tr_loss"] = train_loss
                    best_loss_tr = train_loss
                scheduler.step()
                if classification:
                    print('Train loss: '+str(train_loss)+", test loss: "+str(test_loss)
                        +'test accuracy: ' + np.nanmean(accuracy))
                else:
                    print('Train loss: '+str(train_loss)+", test loss: "+str(test_loss)) 
                # losses_ts.append(per_output_loss.detach().cpu().numpy())
                losses_ts.append(test_loss)
                if not np.isnan(train_loss) and not best_loss_ts or (test_loss < best_loss_ts):
                    if use_wandb:
                        wandb.run.summary["best_loss"] = test_loss
                    best_loss_ts = test_loss
            #print(list(model.parameters())[4])
    return model, best_loss_tr, best_loss_ts, losses_tr, losses_ts

if __name__ == "__main__":
    import argparse
    import wandb

    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('-lr', '--learning_rate', default=.01, type=float)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-e', '--enc_layers', default=2, type=int)
    parser.add_argument('-d', '--dec_layers', default=1, type=int)
    parser.add_argument('-ol', '--output_layers', default=1, type=int)
    parser.add_argument('-u', '--hidden_units', default=64, type=int)
    parser.add_argument('-s', '--step_size', default=60, type=int)
    parser.add_argument('-g', '--gamma', default=.9, type=float)
    parser.add_argument('-f', '--features', default=2, type=int)
    parser.add_argument('-n', '--sample_size', default=1000, type=int)
    parser.add_argument('-eo', '--enc_layers_outer', default=2, type=int)
    parser.add_argument('-do', '--dec_layers_outer', default=1, type=int)
    parser.add_argument('-ei', '--enc_layers_inner', default=2, type=int)
    parser.add_argument('-di', '--dec_layers_inner', default=1, type=int)
    parser.add_argument('-uo', '--hidden_units_outer', default=64, type=int)
    parser.add_argument('-ui', '--hidden_units_inner', default=64, type=int)
    parser.add_argument('--normalization', default="true", type=str)
    parser.add_argument('--connect_decoder', default="true", type=str)

    parser.add_argument('--layer_norm', default='false', type=str)
    parser.add_argument('--batch_norm', default='false', type=str)
    parser.add_argument('--instance_norm', default='false', type=str)
    parser.add_argument('--sample_norm', default='false', type=str)
    parser.add_argument('--mean_center', default='false', type=str)
    parser.add_argument('--quantile_scaling', default='false', type=str)
    parser.add_argument('--seed_weights', default=0, type=int)
    parser.add_argument('-a', '--activation', default='relu', help='relu|elu', type=str)
    parser.add_argument('-m', '--model', default='deepsets', type=str, help='deepsets|settransformer|deepsamples')
    parser.add_argument('--path', default='distribution_plots/', type=str)
    parser.add_argument('--wandb_test', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_size', default=10000, type=int)
    args = parser.parse_args()

    layer_norm = str_to_bool_arg(args.layer_norm, 'layer_norm')
    batch_norm = str_to_bool_arg(args.batch_norm, 'batch_norm')
    instance_norm = str_to_bool_arg(args.instance_norm, 'instance_norm')
    sample_norm = str_to_bool_arg(args.sample_norm, 'sample_norm')
    mean_center = str_to_bool_arg(args.mean_center, 'mean_center')
    ensemble_network = str_to_bool_arg(args.ensemble_network, 'ensemble_network')
    quantile_scaling = str_to_bool_arg(args.quantile_scaling, 'quantile_scaling')
    args.output_name = args.output_name.split()
    normalization = str_to_bool_arg(args.normalization, 'normalization')
    connect_decoder = str_to_bool_arg(args.connect_decoder, 'connect_decoder')
    
    
    wandb.init(project='hematocrit')
    data_config = {
        'inputs': ["rbc", "retics", "plt", "basos", "perox"],
        'outputs': ["Hematocrit"],
        'id_file': "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/id_files/Hematocrit_rbc,retics,plt,basos,perox_January-1-2021.txt",
        'num_samples': 1000,
        'num_subsamples': 1000,
        'permute_subsamples': False,
        'normalizer': "all",
        'imputation': "zero"
    }
    Dataset = FullLargeDataset
    train = Dataset(test=False, **data_config)
    test = Dataset(test=True, **data_config)
    num_workers = 32
    n_dists = 5
    n_final_outputs = 1
    output_names = ['hematocrit']
    
    train_generator = DataLoader(train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    test_generator = DataLoader(test,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=True)

    device = 'cpu' if args.cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if args.activation == 'relu':
        activation = nn.ReLU
    elif args.activation == 'elu':
        activation = nn.ELU

    n_outputs = 1
    if args.model == 'deepsample':
        model = DeepSamples(n_inputs=n_inputs, n_outputs=n_final_outputs, n_enc_layers_outer=args.enc_layers_outer, n_hidden_units_outer=args.hidden_units_outer, n_dec_layers_outer=args.dec_layers_outer, 
        n_enc_layers_inner=args.enc_layers_inner, n_hidden_units_inner=args.hidden_units_inner, n_dec_layers_inner=args.dec_layers_inner,
        activation=activation, normalization=args.normalization=="true", 
        connect_decoder=args.connect_decoder=="true").to(device)
        n_inputs = args.features
    else:
        if args.model == 'settransformer':
            model_unit = SmallSetTransformer
            n_inputs = args.features
        elif args.model == 'deepsets-sum':
            model_unit = BasicDeepSetSum
            n_inputs = args.features
        elif args.model == 'deepsets-rc':
            model_unit = BasicDeepSetMeanRC
            n_inputs = args.features
        else:
            model_unit = BasicDeepSetMean
            n_inputs = args.features
        model = model_unit(n_inputs=n_inputs, n_outputs=n_outputs, n_enc_layers=args.enc_layers, n_hidden_units=args.hidden_units, n_dec_layers=args.dec_layers, ln=layer_norm, bn=batch_norm, activation=activation, instance_norm=instance_norm, n_samples=args.sample_size, sample_norm=sample_norm).to(device)
    
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    output_names = ['hematocrit']
    print(output_names)
    model, train_score, test_score, losses_tr, losses_ts = train_nn(model, 'tentative', optimizer, scheduler, 
                                            train_generator, test_generator, n_epochs=args.epochs,
                                            outputs=output_names, use_wandb=True, plot_gradients=False, seed=args.seed_weights)
