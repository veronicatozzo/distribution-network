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


#from .src.dataset import FullLargeDataset


def plot_moments_distribution(train, outputs_names, path=''):
    X_tr, y_tr, lengths = zip(*[train[i] for i in range(len(train))])
    for i in range(len(outputs_names)):
        aux_x = [y_tr[j][i*2] for j in range(len(train))]
        aux_y = [y_tr[j][i*2+1] for j in range(len(train))]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(aux_x)
        ax[1].hist(aux_y)
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title(outputs_names[i]+' of x axis, Baseline MSE: '+str(mean_squared_error(aux_x, np.ones(10000)*np.mean(aux_x))))
        ax[1].set_title(outputs_names[i]+' of y axis, Baseline MSE: '+str(mean_squared_error(aux_y, np.ones(10000)*np.mean(aux_y))))
        ax[0].axvline(np.mean(aux_x), color='red')
        ax[1].axvline(np.mean(aux_y), color='red')
        plt.tight_layout()
        plt.savefig(path+outputs_names[i]+'.png', dpi=200, bbox_inches='tight')

def get_moment(samples, name):
    if name == 'covariance':
        return empirical_covariance(samples)[0, 1]
    elif name == 'var':
        return np.square(np.std(samples, axis=0))
    else:
        raise ValueError("unknown name: {}".format(name))

def plot_2d_moments_dist_and_func(train, output_names, path=''):
    """ Plot distribution and function

    dist: (x,y,z) is 2d histogram of moments
    func: (x,y,z) is 2d function of moments 
    """
    print('in func')
    assert len(output_names) == 3
    # X_tr, y_tr, lengths = zip(*[train[i] for i in range(len(train))])
    # xs = get_moment(X_tr, output_names[0])
    # ys = get_moment(X_tr, output_names[1])
    # zs = y_tr
    xs = []
    ys = []
    zs = []
    for el in train:
        samples, labels, lengths = el
        xs.append(get_moment(samples, output_names[0]))
        ys.append(get_moment(samples, output_names[1]))
        zs.append(labels)
    xs = np.array(xs)
    ys = np.array(ys)
    # for covariance
    assert len(xs.shape) == 1
    # y is 2 features (will always be the case except for cov)
    assert ys.shape[1] == 2
    y0s = ys[:,0]
    y1s = ys[:,1]
    feats = {'cov': xs, 'var0': y0s, 'var1': y1s}
    plt.hist(zs)
    plt.savefig('cov-var-function.png')
    print(zs.min(), zs.max())
    plt.clf()
    print('mse', mean_squared_error(zs, zs.mean()))
    import sys; sys.exit()
    import seaborn as sns
    print(labels)
    print(np.array(labels).shape)
    print(max(labels) - min(labels))
    print(np.array(labels).std())
    plt.hist(labels)
    plt.xscale('log')
    plt.savefig(os.path.join(path, 'cov-var-function.png'))
    plt.clf()
    for name0, name1 in itertools.combinations(feats.keys(), 2):
        print(name0, name1)
        sns.kdeplot(feats[name0], feats[name1], label='_'.join([name0, name1]))
        plt.legend()
        plt.savefig(os.path.join(path, f'{name0}_{name1}_dist.png'))



class SyntheticDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, n_dim=2, output_names=None, distribution='normal', random_state=0, mean_center=False):
        self.N = N
        self.n_samples = n_samples
        self.n_dim = n_dim
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
            
            if mean_center:
                X = X - np.mean(X, axis=0)
            self.Xs.append(X)
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
            
            # y = [means2.ravel(), means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel()][:n_outputs]
            # y = [np.square(stds.ravel()), means2.ravel(), means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel()][:n_outputs]
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
                    y += [covariances.ravel()]
                elif output_name == 'cov-var-function':
                    y += [np.square(covariances)/2 * logsumexp(stds, axis=0).ravel()]
                elif output_name == 'cov-var':
                    y += [np.square(stds.ravel()), covariances.ravel()]
                # else:
                # y += [means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel(), medians.ravel(), covariances.ravel()][:n_outputs]
                # y += [means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel(), covariances.ravel(), quantiles][:n_outputs]
            #y = [means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel(), covariances.ravel()][:n_outputs]
            y = np.concatenate(y).ravel()
            self.ys.append(y)
        self.Xs = np.array(self.Xs)
        self.ys = np.array(self.ys)

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], np.arange(self.ys[index].shape[0]).reshape(-1, 1)
        
    def __len__(self):
        return self.N


class BasicDeepSet(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=4, n_hidden_units=64, n_dec_layers=1, 
                 multiplication=True,ln=False, bn=False, activation=nn.ReLU, **kwargs):
        super().__init__()
        enc_layers = []
        # enc_layers.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units))
        # for i in range(n_enc_layers - 1):
        #     enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        #     if i < n_enc_layers - 2:
        #         enc_layers.append(activation())
        for i in range(n_enc_layers):
            if i == 0:
                enc_layers.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units))
            else:
                enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            if ln:
                enc_layers.append(nn.LayerNorm(n_hidden_units))
            if bn:
                enc_layers.append(nn.BatchNorm1d(n_hidden_units))
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
                if bn:
                    dec_layers.append(nn.BatchNorm1d(n_hidden_units))
                dec_layers.append(activation())
        self.dec = nn.Sequential(*dec_layers)
        self.multiplication=multiplication

    def forward(self, x):
        if len(x.shape) == 4 and x.shape[1] > 1:
            encoded = []
            for j in range(x.shape[1]):
                a = x[:, j, :, :].squeeze(1)
                encoded.append(self.enc(a))
            x = torch.cat(encoded, 1)
        else:
            x = x.squeeze(1)
            out = self.enc(x)
            #x = torch.mul(x, out)
        return out
    
class BasicDeepSetMean(BasicDeepSet):
    def forward(self, x, length=None):
#         x = super().forward(x)
        x = self.enc(x)
        if self.multiplication:
            x = torch.mul(x, x)
        x = x.mean(dim=-2)
        x = self.dec(x)
        return x

    
class BasicDeepSetSum(BasicDeepSet):
    def forward(self, x, length=None):
#         x = super().forward(x)
        x = self.enc(x)
        if self.multiplication:
            x = torch.mul(x, x)
        x = x.sum(dim=-2)
        x = self.dec(x)
        return x

    
class EnsembleNetwork(nn.Module):
    def __init__(self, models, n_outputs=1, n_hidden_outputs=1, n_inputs=1, n_dist=1, layers=1, multi_input=False, device='cpu:0', activation=nn.ReLU()):
        super(EnsembleNetwork, self).__init__()
        self.models = nn.ModuleList(models)
        self.multi_input = multi_input
        self.n_outputs = n_outputs
        self.device=device
        assert n_inputs > len(models)
        n_o = n_hidden_outputs if multi_input else n_outputs
        if layers == 1:
            self.classifier = nn.Linear(n_inputs, n_o)
        else:
            output_layers = []
            for i in range(layers):
                if i == layers - 1:
                    output_layers.append(nn.Linear(n_inputs, n_o))
                else:
                    output_layers.append(nn.Linear(n_inputs, n_inputs))
                output_layers.append(activation())
            # remove last non-linearity
            output_layers = output_layers[:-1]
            self.classifier = nn.Sequential(*output_layers)
        if self.multi_input:
            self.models_ = [[copy.deepcopy(m) for m in self.models] for i in range(n_dist)]
            self.classifiers_ = [copy.deepcopy(self.classifier) for i in range(n_dist)]
            self.dec = nn.Linear(n_o*n_dist, self.n_outputs)
        
    def forward(self, x, lengths=None):
        if self.multi_input and len(x.shape) == 4 and x.shape[1] > 1:
            multi_output = []
            for j in range(x.shape[1]):
                a = x[:, j, :, :].squeeze(1)
                xs = []
                for m in self.models_[j]:
                    xs.append(m.forward(a.clone().to(self.device)))
                out = torch.cat(xs, dim=-1)
                if len(x.shape) == 3:
                    out = out.reshape((out.shape[0], -1))
                multi_output.append(self.classifiers_[j](out))
            x = torch.cat(multi_output, 1)
            x = self.dec(x)
        else:
            xs = []
            for m in self.models:
                xs.append(m.forward(x.squeeze().clone().to(self.device)))
            x = torch.cat(xs, dim=-1)
            if len(x.shape) == 3:
                x = x.reshape((x.shape[0], -1))
            x = self.classifier(x)
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
    
    # for p in model.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     print(param_norm.item())
    #     print(p.grad.data)
    #     total_norm += param_norm.item() ** 2
    # total_norm = total_norm ** (1. / 2)
    # print('norm', total_norm)

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
    # greater than 1 currently doesn't work
    #parser.add_argument('-o', '--outputs', default=1, type=int)  # total outputs will be outputs * features
    parser.add_argument('-om', '--output_multiplier', default=1, type=int)  # total features before linear layer will be outputs * features * om
    # parser.add_argument('-on', '--output_name', metavar='N', type=str, nargs='+',
    #                     help='a list of strings denoting the output types')
    # since wandb can't handle multi-outputs; to use on multiple outputs: -on "mean var cov"
    parser.add_argument('-on', '--output_name', type=str,
                        help='a list of strings denoting the output types')
    parser.add_argument('--name', type=str)
    parser.add_argument('--hematocrit', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--layer_norm', default='false', type=str)
    parser.add_argument('--batch_norm', default='false', type=str)
    parser.add_argument('--mean_center', default='false', type=str)
    parser.add_argument('--seed_weights', default=0, type=int)
    parser.add_argument('--seed_dataset', default=0, type=int)
    parser.add_argument('--distribution', default='normal', help='normal|gamma|t', type=str)
    parser.add_argument('-a', '--activation', default='relu', help='relu|elu', type=str)
    parser.add_argument('-m', '--model', default='deepsets', type=str, help='deepsets|settransformer|deepsamples')
    parser.add_argument('--ensemble_network', default='false', type=str)
    parser.add_argument('--path', default='distribution_plots/', type=str)
    parser.add_argument('--wandb_test', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_size', default=10000, type=int)
    args = parser.parse_args()

    layer_norm = str_to_bool_arg(args.layer_norm, 'layer_norm')
    batch_norm = str_to_bool_arg(args.batch_norm, 'batch_norm')
    mean_center = str_to_bool_arg(args.mean_center, 'mean_center')
    ensemble_network = str_to_bool_arg(args.ensemble_network, 'ensemble_network')
    if not args.wandb_test:
        args.output_name = args.output_name.split()
    
    if args.wandb_test:
        wandb.init(project='wandb_test')
    else:
        if args.name:
            wandb.init(project='synthetic-moments1', name=args.name)
        else:
            wandb.init(project='synthetic-moments1')
    if args.hematocrit:
        data_config = {
            'inputs': ["rbc", "retics", "plt", "basos", "perox"],
            'outputs': ["Hematocrit"],
            'id_file': "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/id_files/Hematocrit_rbc,retics,plt,basos,perox_January-1-2021.txt",
            # 'num_samples': args.num_samples,
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
        n_final_outputs = args.outputs
        output_names = ['hematocrit']
    else:
        train = SyntheticDataset(args.train_size, args.sample_size, args.features, args.output_name, args.distribution, args.seed_dataset, mean_center)
        # standardscaler = StandardScaler()
        # X = standardscaler.fit_transform(train.Xs.reshape((-1, train.Xs.shape[-1]))).reshape(train.Xs.shape)
        # train.Xs = X
        # X = train.Xs.reshape((-1, train.Xs.shape[-1]))
        # quantile_scaler = QuantileScaler()
        # X_new = quantile_scaler.fit_transform(X)
        # train.Xs = X_new.reshape(train.Xs.shape)
        test = SyntheticDataset(1000, args.sample_size, args.features,args.output_name, args.distribution, args.seed_dataset, mean_center)
        # X = standardscaler.transform(test.Xs.reshape((-1, test.Xs.shape[-1]))).reshape(test.Xs.shape)
        # test.Xs = X
        # X = test.Xs.reshape((-1, test.Xs.shape[-1]))
        # X_new = quantile_scaler.transform(X)
        # test.Xs = X_new.reshape(test.Xs.shape)

        num_workers = 1
        n_dists = 1
        if args.output_name == ['cov-var-function']:
            n_final_outputs = 1
        elif args.output_name == ['cov-var']:
            n_final_outputs = 3
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
    device = 'cpu' if args.cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model == 'settransformer':
        model_unit = SmallSetTransformer
        n_inputs = args.features
    elif args.model == 'deepsamples':
        model_unit = SmallDeepSamples
        # n_inputs for deepsamples should actually be n_dists
        # we code in this way to maintain compatibility with main.py
        n_inputs = 1
    elif args.model == 'deepsets-sum':
        model_unit = BasicDeepSetSum
        n_inputs = args.features
    else:
        model_unit = BasicDeepSetMean
        n_inputs = args.features
    
    if args.activation == 'relu':
        activation = nn.ReLU
    elif args.activation == 'elu':
        activation = nn.ELU

    n_outputs = 0
    for output_name in args.output_name:
        if output_name in ['cov', 'cov-var-function']:
            n_outputs += 1
        else:
            n_outputs += args.features
    
    if ensemble_network:
        model = EnsembleNetwork([model_unit(n_inputs=n_inputs, n_outputs=args.features, n_enc_layers=args.enc_layers, n_hidden_units=args.hidden_units, n_dec_layers=args.dec_layers, ln=layer_norm, bn=batch_norm, activation=activation).to(device) 
                            for i in range(num_models)], n_outputs=n_outputs, device=device, layers=args.output_layers, n_inputs=num_models * args.features * n_dists)
    else:
        model = model_unit(n_inputs=n_inputs, n_outputs=n_outputs, n_enc_layers=args.enc_layers, n_hidden_units=args.hidden_units, n_dec_layers=args.dec_layers, ln=layer_norm, bn=batch_norm, activation=activation).to(device)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    # output_names = list(itertools.product(['E(x^2) - E(x)^2', 'E(x^2)', 'E(x)', 'std', 'skew', 'kurtosis'][:args.outputs], range(args.features)))
    output_names = list(map(str, itertools.product(args.output_name, range(args.features))))
    # cov must be last in the list
    if 'cov' in args.output_name:
        output_names = output_names[:-1]
    # only one output, not one per feature
    elif args.output_name == ['cov-var-function']:
        output_names = [args.output_name]
    elif args.output_name == ['cov-var']:
        output_names = ['var0', 'var1', 'cov']
    else:
        output_names = list(map(str, itertools.product(args.output_name, range(args.features))))
    model, train_score, test_score, losses_tr, losses_ts = train_nn(model, 'tentative', optimizer, scheduler, 
                                            train_generator, test_generator, n_epochs=args.epochs,
                                            outputs=output_names, use_wandb=True, plot_gradients=False, seed=args.seed_weights)
