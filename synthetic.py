from scipy.stats import kurtosis, skew
from torch.utils.data import Dataset
from sklearn.datasets import make_spd_matrix
from torch.utils.data import DataLoader
import numpy as np
import itertools
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy

#from .src.dataset import FullLargeDataset


class SyntheticDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, n_distr=1, n_dim=2, n_outputs=1, output_name='x^2'):
        self.N = N
        self.n_samples = n_samples
        self.n_dim = n_dim
        self.Xs = []
        self.ys = []
        for n in range(N):
            x = []
            y = []
            for d in range(n_distr):
                cov = make_spd_matrix(self.n_dim)
                X = np.random.multivariate_normal(np.random.randn(self.n_dim), cov, size=self.n_samples, check_valid='warn', tol=1e-8)
                x.append(X)
                X2 = X**2
                means2 = np.mean(X2, axis=0)
                means = np.mean(X, axis=0)
                stds = np.std(X, axis=0)

                skews = skew(X, axis=0)
                kurtoses = kurtosis(X, axis=0)
                if self.n_dim > 1:
                    covariances = np.array(cov[0, 1])
                quantiles = np.quantile(X, np.arange(.1, 1, .1), axis=0).ravel()
                output_names = [output_name]
                # y = [means2.ravel(), means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel()][:n_outputs]
                # y = [np.square(stds.ravel()), means2.ravel(), means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel()][:n_outputs]
                if output_name == 'mean':
                    y += [means.ravel()]
                elif output_name == 'x^2':
                    y += [means2.ravel()]
                elif output_name == 'var':
                    y += [np.square(stds.ravel())]
                elif output_name == 'skew':
                    y += [skews.ravel()]
                elif output_name == "kurtosis":
                    y += [kurtosis.ravel()]
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
                else:
                    y += [means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel(), covariances.ravel(), quantiles][:n_outputs]
            #y = [means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel(), covariances.ravel()][:n_outputs]
            # print('before', np.array(y).shape)
            y = np.concatenate(y).ravel()
            self.Xs.append(np.array(x))
            self.ys.append(y)
            # self.ys.append(np.concatenate([means2.ravel()]))  #, means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel()]).ravel())
            #, means.ravel(),stds.ravel(), skews.ravel(), kurtoses.ravel(), covariances.ravel()]).ravel())

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], np.arange(1).reshape(-1, 1)
        
    def __len__(self):
        return self.N


class BasicDeepSet(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=4, n_hidden_units=64, n_dec_layers=1, 
                 multiplication=True,**kwargs):
        super().__init__()
        enc_layers = []
        # enc_layers.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units))
        # for i in range(n_enc_layers - 1):
        #     enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        #     if i < n_enc_layers - 2:
        #         enc_layers.append(nn.ReLU())
        for i in range(n_enc_layers):
            if i == 0:
                enc_layers.append(nn.Linear(in_features=n_inputs, out_features=n_hidden_units))
            else:
                enc_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            enc_layers.append(nn.ReLU())
        # remove last relu
        enc_layers = enc_layers[:-1]
        self.enc = nn.Sequential(*enc_layers)
        dec_layers = []
        # for i in range(n_dec_layers - 1):
        #     dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        #     dec_layers.append(nn.ReLU())
        # dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for i in range(n_dec_layers):
            if i == n_dec_layers - 1:
                dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
            else:
                dec_layers.append(nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
            dec_layers.append(nn.ReLU())
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
    def __init__(self, models, n_outputs=1, n_hidden_outputs=1, n_inputs=1, n_dist=1, layers=1, multi_input=False, device='cpu:0'):
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
                output_layers.append(nn.ReLU())
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
             n_epochs=10, outputs=[], use_wandb=True, plot_gradients=True):
    if use_wandb:
        import wandb
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(classification)
    model = model.to(device)
    if use_wandb:
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
            #print(x.shape, y.shape)
            x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
            loss_elements = criterion(model(x, lengths), y)
            loss = loss_elements.mean()
            if np.isnan(loss.item()):
                raise ValueError("Train loss is nan: ", loss)
            train_aux.append(loss.item())
            # TODO: maybe we don't want to log at every step
            if use_wandb:
                wandb.log({f"{name} train loss per step": loss}, step=step)
            if len(outputs) > 1:
                outputs_loss = loss_elements.mean(dim=0)
                assert len(outputs) == len(outputs_loss)
                per_output_loss = {o: l for o, l in zip(outputs, outputs_loss)}
                if use_wandb:
                    wandb.log({f"{name} train loss per step, stratified": per_output_loss}, step=step)
            else:
                per_output_loss = {0: loss}
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                losses_tr.append(per_output_loss)
                
                aux = []
                accuracy = []
                for x, y, lengths in test_generator:
                    x, y, lengths = x.type(dtype).to(device), y.type(dtype).to(device), lengths.to(device)
                    loss_elements = criterion(model(x, lengths), y)
                    loss = loss_elements.mean()
                    if np.isnan(loss.item()):
                        raise ValueError("Test loss is nan: ", loss)
                    if classification:
                        accuracy.append(accuracy_score(model(x, lengths).detach().cpu().numpy(),
                                                       y.detach().cpu().numpy().astype(np.int8)))
                    aux.append(loss.item())
                test_loss = np.nanmean(aux)
                if use_wandb:
                    wandb.log({f"{name} test loss per step": test_loss}, step=step)
                if len(outputs) > 1:
                    outputs_loss = loss_elements.mean(dim=0)
                    assert len(outputs) == len(outputs_loss)
                    per_output_loss = {o: l for o, l in zip(outputs, outputs_loss)}
                    if use_wandb:
                        wandb.log({f"{name} train loss per step, stratified": per_output_loss}, step=step)
                else:
                     per_output_loss = {0: loss}
                train_loss = np.nanmean(train_aux)
                print(train_loss)
                if not np.isnan(train_loss) and not best_loss_tr or (train_loss < best_loss_tr):
                    if use_wandb:
                        wandb.run.summary["best_loss"] = train_loss
                    best_loss_tr = train_loss
                scheduler.step()
                if classification:
                    print('Train loss: '+str(train_loss)+", test loss: "+str(test_loss)
                        +'test accuracy: ' + np.nanmean(accuracy))
                else:
                    print('Train loss: '+str(train_loss)+", test loss: "+str(test_loss)) 
                losses_ts.append(per_output_loss)
                if not np.isnan(train_loss) and not best_loss_ts or (test_loss < best_loss_ts):
                    if use_wandb:
                        wandb.run.summary["best_loss"] = test_loss
                    best_loss_ts = test_loss
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
    parser.add_argument('-o', '--outputs', default=1, type=int)  # total outputs will be outputs * features
    parser.add_argument('-om', '--output_multiplier', default=1, type=int)  # total features before linear layer will be outputs * features * om
    parser.add_argument('-on', '--output_name', default='x^2', help='x^2|var', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--hematocrit', action='store_true')
    args = parser.parse_args()

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
            'normalizer': "none",
            'imputation': "zero"
        }
        Dataset = FullLargeDataset
        train = Dataset(test=False, **data_config)
        test = Dataset(test=True, **data_config)
        num_workers = 4
        n_dists = 5
        n_final_outputs = args.outputs
    else:
        train = SyntheticDataset(10000, args.sample_size, args.features, args.outputs, args.output_name)
        test = SyntheticDataset(1000, args.sample_size, args.features, args.outputs, args.output_name)
        num_workers = 1
        n_dists = 1
        n_outputs = args.outputs
        n_final_outputs = n_outputs * args.features if n_outputs < 5 else n_outputs * args.features - 1
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

    n_outputs = args.outputs
    num_models = n_outputs * args.output_multiplier
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EnsembleNetwork([BasicDeepSetMean(n_inputs=args.features, n_outputs=args.features, n_enc_layers=args.enc_layers, n_hidden_units=args.hidden_units, n_dec_layers=args.dec_layers).to(device) 
                            for i in range(num_models)], n_outputs=n_final_outputs, device=device, layers=args.output_layers, n_inputs=num_models * args.features * n_dists)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    # output_names = list(itertools.product(['E(x^2) - E(x)^2', 'E(x^2)', 'E(x)', 'std', 'skew', 'kurtosis'][:args.outputs], range(args.features)))
    if args.output_name == 'all':
        output_names = list(itertools.product(['mean', 'std', 'skew', 'kurtosis', 'covariance'][:args.outputs], range(args.features)))
    else:
        output_names = list(itertools.product([args.output_name], range(args.features)))
    # covariance only has one
    if args.outputs == 5:
        output_names = output_names[:-1]
    model, train_score, test_score, losses_tr, losses_ts = train_nn(model, 'tentative', optimizer, scheduler, 
                                            train_generator, test_generator, n_epochs=100,
                                            outputs=output_names, use_wandb=True)