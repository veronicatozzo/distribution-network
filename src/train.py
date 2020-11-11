import os
import warnings 

import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import kurtosis, skew
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from skl_groups.divergences import KNNDivergenceEstimator

from skl_groups.kernels import PairwisePicker, Symmetrize, RBFize, ProjectPSD


def train_nn(model, name, optimizer, scheduler, train_generator, test_generator):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # by default, reduction = mean when multiple outputs
    criterion = nn.MSELoss()
    step = 0
    best_loss = None
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for epoch in range(1000):
        aux = []
        for x, y in train_generator:
            x, y = x.type(dtype).to(device), y.type(dtype).to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} train loss per step": loss}, step=step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        scheduler.step()
        aux = []
        for x, y in test_generator:
            x, y = x.type(dtype).to(device), y.type(dtype).to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} test loss per step": loss}, step=step)
        test_loss = np.mean(aux)
        if not best_loss or (test_loss > best_loss):
            wandb.run.summary["best_loss"] = test_loss
            best_loss = test_loss
    return model




def train_KNNDivergence(divergence, X_tr,  y_tr, X_ts, y_ts, k=5, C=1, name=''):
    """
    Parameters
    ----------
    divergence: string, 
        Type of divergence to use when estimating distance among distribution. 
        Options 'kl','renyi:.8','tsallis:.8','hellinger','bc','l2','linear', 'jensen-shannon'.
    X_tr: array-like
        Training data
    y_tr: array-like
        Training output
    X_ts: array-like
        Test data
    y_ts: array-like
        Test output
    k: int, optional default=5
        Number of k-nearest niehgbours to use for the estimation of the distances. 

    C: float, optional default=1
        Regularization parameter for SVM.
    """
    warnings.simplefilter('ignore')

    pipeline = [
        ('divs', KNNDivergenceEstimator(div_funcs=[divergence], Ks=[k])),
        ('pick', PairwisePicker((0, 0))),
        ('symmetrize', Symmetrize()),
        ('rbf', RBFize(gamma=1, scale_by_median=True)),
        ('project', ProjectPSD()),
    ]
    classification = isinstance(y_tr[0][0], str) or isinstance(y_tr[0][0], bool)
    if classification:
        pipeline.append(('svm', SVC(C=C, kernel='precomputed')),)
    else:
        pipeline.append(('svm', SVR(C=C, kernel='precomputed')),)
    model = Pipeline(pipeline)
    print('y_ts', y_ts)
    X_tr = [x for x in X_tr]
    y_tr = [y for y in y_tr]
    X_ts = [x for x in X_ts]
    y_ts = [y for y in y_ts]
    # X_tr = list(X_tr)
    # y_tr - list(y_tr)
    # X_ts = list(X_ts)
    # y_ts = list(y_ts)
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_ts)
    pd.DataFrame.from_dict({'preds': preds, 'labels': y_ts}).to_csv(name + '.csv')
    if classification:
        train_score = accuracy_score(y_tr, model.predict(X_tr))
        test_score = accuracy_score(y_ts, preds)
    else:
        train_score = mean_squared_error(y_tr, model.predict(X_tr))
        test_score = mean_squared_error(y_ts, model.predict(X_ts))
    # wandb.log({'train_mse': train_score, 'test_mse': test_score})
    #print(train_score, test_score)
    return train_score, test_score


def train_distribution2distrbution(X_tr,  y_tr, X_ts, y_ts):
    import matlab.engine
    import matlab
    eng = matlab.engine.start_matlab()

    path = os.path.realpath(__file__).split('/')[:-2]
    path ='/'.join(path) + '/distribution2distribution'
    eng.addpath(eng.genpath(path), nargout=0)
    print(X_tr.shape)
    x_tr = [matlab.double(x.tolist()) for x in X_tr]
    y_tr_ = matlab.double([[y] for y in y_tr])
    osp = eng.osde(x_tr)
    PCin = osp['pc']
    basis_inds = osp['inds']    

    [B, rks, tst_stats, cv_stats] = eng.rks_ridge(PCin, y_tr_, nargout=4)
    y_tr_est = eng.predict(PCin, rks, B)

    x_ts = [matlab.double(x.tolist()) for x in X_ts]
    osp = eng.osde(x_ts, {'inds': basis_inds})
    PCpred = osp['pc']
    y_ts_est = eng.predict(PCpred, rks, B)

    train_score = mean_squared_error(y_tr, np.array(y_tr_est))
    test_score = mean_squared_error(y_ts, np.array(y_ts_est))
    # wandb.log({'train_mse': train_score, 'test_mse': test_score})


    return train_score, test_score


def get_moments(X):
    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    skews = skew(X, axis=1)
    kurtoses = kurtosis(X, axis=1)
    return np.concatenate([means, stds, skews, kurtoses], axis=1)


def train_KNNMoments(X_tr,  y_tr, X_ts, y_ts, k=5, name=''):
    X_tr = get_moments(X_tr)
    X_ts = get_moments(X_ts)
    classification = isinstance(y_tr[0][0], str) or isinstance(y_tr[0][0], bool)
    if classification:
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_ts)
    pd.DataFrame.from_dict({'preds': preds, 'labels': y_ts}).to_csv(name + '.csv')
    if classification:
        train_score = accuracy_score(y_tr, model.predict(X_tr))
        test_score = accuracy_score(y_ts, preds)
    else:
        train_score = mean_squared_error(y_tr, model.predict(X_tr))
        test_score = mean_squared_error(y_ts, model.predict(X_ts))
    return train_score, test_score