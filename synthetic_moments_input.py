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

class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden_units, n_layers):
        super(MLP, self).__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(n_inputs, 1))
        else:
            for i in range(n_layers):
                if i == 0:
                    layers.append(nn.Linear(n_inputs, n_hidden_units))
                elif i == n_layers - 1:
                    layers.append(nn.Linear(n_hidden_units, 1))
                else:
                    layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                layers.append(nn.ReLU())
            layers = layers[:-1]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x, length=None):
        return self.layers(x)
        # return self.layers(x).reshape((-1, 1, 1, 1))

class SyntheticMomentsDataset(Dataset):
    def __init__(self, N=1000, n_samples=500, n_dim=2, output_names=None, distribution='normal', random_state=0):
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
            stds = np.std(X, axis=0)
            covariances = np.array(empirical_covariance(X)[0, 1]).reshape(1, 1)
            moments = [np.square(stds.ravel()), covariances.ravel()]
            moments = np.concatenate(moments).ravel()
            self.Xs.append(np.array(moments))
            y = [np.square(covariances)/2 * logsumexp(stds, axis=0).ravel()]
            self.ys.append(np.array(y))
        # print('before', np.array(self.ys).shape)
        self.ys = np.array(self.ys).reshape((-1, 1))
        # print('after', self.ys.shape)
    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], np.arange(len(self.ys[index])).reshape(-1, 1)
        
    def __len__(self):
        return self.N


if __name__ == "__main__":
    import argparse
    import wandb

    parser = argparse.ArgumentParser(description='Results summary')
    parser.add_argument('-lr', '--learning_rate', default=.01, type=float)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-ol', '--output_layers', default=1, type=int)
    parser.add_argument('-u', '--hidden_units', default=64, type=int)
    parser.add_argument('-s', '--step_size', default=60, type=int)
    parser.add_argument('-g', '--gamma', default=.9, type=float)
    parser.add_argument('-f', '--features', default=2, type=int)
    parser.add_argument('-n', '--sample_size', default=1000, type=int)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed_weights', default=0, type=int)
    parser.add_argument('--seed_dataset', default=0, type=int)
    parser.add_argument('--distribution', default='normal', help='normal|gamma|t', type=str)
    parser.add_argument('--wandb_test', action='store_true')
    parser.add_argument('--cpu', default='store_true')
    # dummy, placeholder for future
    parser.add_argument('--output_name', default='cov-var-function')
    parser.add_argument('--epochs', default=100, type=int)
    args = parser.parse_args()
    
    if args.wandb_test:
        wandb.init(project='wandb_test')
    else:
        if args.name:
            wandb.init(project='synthetic-moments1', name=args.name)
        else:
            wandb.init(project='synthetic-moments1')
    
    train = SyntheticMomentsDataset(10000, args.sample_size, args.features, args.output_name, args.distribution, args.seed_dataset)
    test = SyntheticMomentsDataset(1000, args.sample_size, args.features,args.output_name, args.distribution, args.seed_dataset)

    model = MLP(3, args.hidden_units, args.output_layers)
    # print(model)
    # import sys; sys.exit()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    from sklearn.linear_model import RidgeCV

    mdl = RidgeCV()
    X_tr = np.array([train[i][0] for i in range(len(train))])
    print(X_tr.shape)
    Y_tr = np.squeeze(np.array([train[i][1] for i in range(len(train))]), axis=(-1, -2))
    print(Y_tr.shape)
    X_ts = np.array([test[i][0] for i in range(len(test))])
    print(X_ts.shape)
    Y_ts = np.squeeze(np.array([test[i][1] for i in range(len(test))]), axis=(-1, -2))
    print(Y_ts.shape)
    mdl.fit(X_tr, Y_tr)
    ridge_loss = mean_squared_error(Y_ts, mdl.predict(X_ts))
    ridge_tr_loss = mean_squared_error(Y_tr, mdl.predict(X_tr))
    print('test error: ', ridge_loss)
    print('train error: ', ridge_tr_loss)
    wandb.run.summary["ridge_tr_loss"] = ridge_tr_loss
    wandb.run.summary["ridge_loss"] = ridge_loss

    num_workers = 32
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
    
    model, train_score, test_score, losses_tr, losses_ts = train_nn(model, 'tentative', optimizer, scheduler, 
                                            train_generator, test_generator, n_epochs=args.epochs,
                                            outputs=['cov-var-function'], use_wandb=True, plot_gradients=False, seed=args.seed_weights)
    

    
