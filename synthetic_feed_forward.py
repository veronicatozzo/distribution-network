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


#from .src.dataset import FullLargeDataset
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.datasets import make_spd_matrix
from sklearn.covariance import empirical_covariance
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import numpy as np
from synthetic import train_nn
from scipy.special import logsumexp


class MLP_aggr(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_samples=1000, n_enc_layers=4, n_hidden_units=64, n_dec_layers=1):
        super().__init__()
        enc_layers = []
        for i in range(n_enc_layers):
            if i == 0:
                enc_layers.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units))
            else:
                enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            enc_layers.append(nn.ReLU())
        enc_layers = enc_layers[:-1]
        self.enc = nn.Sequential(*enc_layers)

        self.aggregation = nn.Linear(in_features=n_samples, out_features=1)

        dec_layers = []
        for i in range(n_dec_layers):
            if i == n_dec_layers - 1:
                dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
            else:
                dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            dec_layers.append(nn.ReLU())
        self.dec = nn.Sequential(*dec_layers)
        
    def forward(self, x, length=None):
        out = self.enc(x)
        out = torch.transpose(out, 1, 2)
        out = self.aggregation(out)
        out = out.squeeze()
        out = self.dec(out)
        return out
        # return self.layers(x).reshape((-1, 1, 1, 1))


class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden_units, n_layers, n_outputs):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(n_inputs, 1))
        else:
            for i in range(n_layers):
                if i == 0:
                    layers.append(nn.Linear(n_inputs, n_hidden_units))
                elif i == n_layers - 1:
                    layers.append(nn.Linear(n_hidden_units, n_outputs))
                else:
                    layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                layers.append(nn.ReLU())
            layers = layers[:-1]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x, length=None):
        
        return self.layers(x).squeeze()


class SyntheticDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, n_dim=2, flatten=False, reshuffling=10, output_names=None, distribution='normal', random_state=0):
        self.N = N
        self.n_samples = n_samples
        self.reshuffling = reshuffling
        self.n_dim = n_dim
        self.flatten = flatten
        self.Xs = []
        self.ys = []
        print('Dataset output', output_names)
        for n in range(N):
            x = []
            y = []
            #for d in range(n_distr):
            if distribution == "normal":
                cov = make_spd_matrix(self.n_dim)
                X = np.random.RandomState(random_state).multivariate_normal(np.random.randn(self.n_dim), cov, size=self.n_samples, check_valid='warn', tol=1e-8)
            elif distribution == "t":
                X = np.random.RandomState(random_state).standard_t(np.random.randint(10, 20, size=self.n_dim), size=(self.n_samples, self.n_dim))
            elif distribution == "gamma":
                X = np.random.RandomState(random_state).gamma(np.random.randint(1, 30, size=self.n_dim), np.random.randint(1, 30, size=self.n_dim), size=(self.n_samples, self.n_dim))
            
            X2 = X**2
            means2 = np.mean(X2, axis=0)
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            medians = np.median(X, axis=0)

            skews = skew(X, axis=0)
            kurtoses = kurtosis(X, axis=0)
            quantiles = np.quantile(X, np.arange(.1, 1, .1), axis=0).ravel()
            
            if self.n_dim > 1:
                covariances = np.array(empirical_covariance(X)[0, 1]).reshape(1, 1)
            quantiles = np.quantile(X, np.arange(.1, 1, .1), axis=0).ravel()
            
            for output_name in output_names:
                if output_name == 'mean':
                    y += [means.ravel()]
                elif output_name == 'x^2':
                    y += [means2.ravel()]
                elif output_name == 'x^3':
                    means3 = np.mean(X**3, axis=0)
                    y = [means3.ravel()]
                elif output_name == 'x^4':
                    means4 = np.mean(X**4, axis=0)
                    y = [means4.ravel()]
                elif output_name == 'var':
                    y += [np.square(stds.ravel())]
                elif output_name == 'skew':
                    y += [skews.ravel()]
                elif output_name == "kurtosis":
                    y += [kurtoses.ravel()]
                elif output_name == 'quantiles_0.1':
                    y += [quantiles[:2]]
                elif output_name == 'quantiles_0.2':
                    y += [quantiles[2:4]]
                elif output_name == 'quantiles_0.3':
                    y += [quantiles[4:6]]
                elif output_name == 'quantiles_0.4':
                    y += [quantiles[6:8]]
                elif output_name == 'quantiles_0.5':
                    y += [quantiles[8:10]]
                elif output_name == 'quantiles_0.6':
                    y += [quantiles[10:12]]
                elif output_name == 'quantiles_0.7':
                    y += [quantiles[12:14]]
                elif output_name == 'quantiles_0.8':
                    y += [quantiles[14:16]]
                elif output_name == 'quantiles_0.9':
                    y += [quantiles[16:18]]
                elif output_name == 'cov':
                    y += [covariances]
                elif output_name == 'cov-var-function':
                    y = [np.square(covariances)/2 * logsumexp(stds, axis=0).ravel()]
            y = np.concatenate(y).ravel()
            
            if flatten:
                if reshuffling:
                    for i in range(reshuffling):
                        out = X.reshape(1, n_samples*n_dim)
                        np.random.shuffle(out)
                        self.Xs.append(out)
                        self.ys.append(y)
                else:
                    self.Xs.append(X.reshape(1, n_samples*n_dim))
                    self.ys.append(y)
            else:
                if reshuffling:
                    for i in range(reshuffling):
                        np.random.shuffle(X)
                        self.Xs.append(X)
                        self.ys.append(y)
                else:
                    self.Xs.append(X)
                    self.ys.append(y)
        self.Xs = np.array(self.Xs)
        self.ys = np.array(self.ys)

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], np.arange(self.ys[index].shape[0]).reshape(-1, 1)
        
    def __len__(self):
        if self.flatten:
            return self.N
        else:
            return self.N*self.reshuffling




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
            x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
            loss_elements = criterion(model(x, lengths), y)
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
    # greater than 1 currently doesn't work
    #parser.add_argument('-o', '--outputs', default=1, type=int)  # total outputs will be outputs * features
    parser.add_argument('-om', '--output_multiplier', default=1, type=int)  # total features before linear layer will be outputs * features * om
    parser.add_argument('-on', '--output_name', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the output types')
    parser.add_argument('--name', type=str)
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--layer_norm', action='store_true')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--seed_weights', default=0, type=int)
    parser.add_argument('--seed_dataset', default=0, type=int)
    parser.add_argument('--reshuffling', default=10, type=int)
    parser.add_argument('--distribution', default='normal', help='normal|gamma|t', type=str)
    parser.add_argument('--path', default='distribution_plots/', type=str)
    parser.add_argument('--wandb_test', action='store_true')
    args = parser.parse_args()
    
    if args.wandb_test:
        wandb.init(project='wandb_test')
    else:
        if args.name:
            wandb.init(project='synthetic-moments1', name=args.name)
        else:
            wandb.init(project='synthetic-moments1')
    train = SyntheticDataset(10000, args.sample_size, args.features, args.flatten, args.reshuffling, args.output_name, args.distribution, args.seed_dataset)
    standardscaler = StandardScaler()
    X = standardscaler.fit_transform(train.Xs.reshape((-1, train.Xs.shape[-1]))).reshape(train.Xs.shape)
    train.Xs = X
    test = SyntheticDataset(1000, args.sample_size, args.features, args.flatten, args.reshuffling, args.output_name, args.distribution, args.seed_dataset)
    X = standardscaler.transform(test.Xs.reshape((-1, test.Xs.shape[-1]))).reshape(test.Xs.shape)
    test.Xs = X
    num_workers = 1
    n_dists = 1
    if args.output_name == ['cov-var-function']:
        n_final_outputs = 1
    else:
        n_final_outputs = len(args.output_name) * args.features if 'cov' not in args.output_name else len(args.output_name)  * args.features - 1
    # output_names = list(itertools.product(['E(x^2) - E(x)^2', 'E(x^2)', 'E(x)', 'std', 'skew', 'kurtosis'][:args.outputs], range(args.features)))
    output_names = list(map(str, itertools.product(args.output_name, range(args.features))))
    # covariance only has one
    # if args.outputs > 5:
    #    output_names = output_names[:-1]
    if args.plot:
        os.makedirs(args.path, exist_ok=True)
        # plot_moments_distribution(train, output_names, path=args.path) # possibly we might want to add something relative to the experiments
        if args.output_name == ["cov-var-function"]:
            plot_2d_moments_dist_and_func(train, ['covariance', 'var', args.output_name], path=args.path)
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

    n_outputs = len(args.output_name)
    num_models = n_outputs * args.output_multiplier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.flatten:
        model = MLP(n_inputs=args.features*args.sample_size, n_hidden_units=args.hidden_units, n_layers=args.enc_layers, n_outputs=n_final_outputs)
    else:
        model =  MLP_aggr(n_inputs=args.features, n_outputs=n_final_outputs, n_samples=args.sample_size, n_enc_layers=args.enc_layers, 
                    n_hidden_units=args.hidden_units, n_dec_layers=args.dec_layers)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    # output_names = list(itertools.product(['E(x^2) - E(x)^2', 'E(x^2)', 'E(x)', 'std', 'skew', 'kurtosis'][:args.outputs], range(args.features)))
    output_names = list(map(str, itertools.product(args.output_name, range(args.features))))
    if 'cov' in args.output_name:
            output_names = output_names[:-1]
    # only one output, not one per feature
    elif args.output_name == ['cov-var-function']:
        output_names = [args.output_name]
    else:
        output_names = list(map(str, itertools.product(args.output_name, range(args.features))))
    model, train_score, test_score, losses_tr, losses_ts = train_nn(model, 'tentative', optimizer, scheduler, 
                                            train_generator, test_generator, n_epochs=100,
                                            outputs=output_names, use_wandb=True, plot_gradients=True, seed=args.seed_weights)
