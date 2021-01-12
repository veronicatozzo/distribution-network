from argparse import ArgumentError
import os
import warnings
from itertools import combinations
import joblib as jl

import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import kurtosis, skew
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.inspection import plot_partial_dependence
from sklearn.impute import SimpleImputer
from memory_profiler import profile


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
    from skl_groups.divergences import KNNDivergenceEstimator

    from skl_groups.kernels import PairwisePicker, Symmetrize, RBFize, ProjectPSD


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
    classification = isinstance(y_tr[0][0], str) or isinstance(y_tr[0][0], bool) or isinstance(y_tr[0][0], np.bool_)
    if classification:
        pipeline.append(('svm', SVC(C=C, kernel='precomputed')),)
    else:
        pipeline.append(('svm', SVR(C=C, kernel='precomputed')),)
    model = Pipeline(pipeline)
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
    pd.DataFrame.from_dict({'preds': preds, 'labels': np.array(y_ts).flatten()}).to_csv(name + '.csv')
    if classification:
        train_score = accuracy_score(np.array(y_tr).flatten(), model.predict(X_tr))
        test_score = accuracy_score(np.array(y_ts).flatten(), preds)
    else:
        train_score = mean_squared_error(y_tr, model.predict(X_tr))
        test_score = mean_squared_error(y_ts, model.predict(X_ts))
    # wandb.log({'train_mse': train_score, 'test_mse': test_score})
    #print(train_score, test_score)
    return train_score, test_score


def train_distribution2distrbution(X_tr,  y_tr, X_ts, y_ts, name=''):
    import matlab.engine
    import matlab
    eng = matlab.engine.start_matlab()

    path = os.path.realpath(__file__).split('/')[:-2]
    path ='/'.join(path) + '/distribution2distribution'
    eng.addpath(eng.genpath(path), nargout=0)
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
    pd.DataFrame.from_dict({'preds': y_ts_est, 'labels': y_ts}).to_csv(name + '.csv')
    # wandb.log({'train_mse': train_score, 'test_mse': test_score})


    return train_score, test_score


# def get_moments(X):
#     means = np.mean(X, axis=1)
#     stds = np.std(X, axis=1)
#     skews = skew(X, axis=1)
#     kurtoses = kurtosis(X, axis=1)
#     return np.concatenate([means, stds, skews, kurtoses], axis=1)

# def get_moments(X):
#     """ We assume X is not a numpy array since num_samples can be different per example """
#     means = [np.mean(samples, axis=0) for samples in X]
#     stds = [np.std(samples, axis=0) for samples in X]
#     skews = [skew(samples, axis=0) for samples in X]
#     kurtoses = [kurtosis(samples, axis=0) for samples in X]
#     covariances = np.array([np.cov(samples, rowvar=False)[0][1] for samples in X]).reshape(-1, 1)
#     print(means[0].shape, stds[0].shape, skews[0].shape, kurtoses[0].shape, covariances[0].shape)
#     print(len(means), len(stds), len(skews), len(kurtoses), len(covariances))
#     return np.concatenate([means, stds, skews, kurtoses, covariances], axis=1)

def get_moments(X):
    """
    X: [n_patients, n_dists, n_samples, 2]
    """
    means = np.array([np.mean(x, axis=1) for x in X]).reshape(len(X), -1)
    stds = np.array([np.std(x, axis=1) for x in X]).reshape(len(X), -1)
    skews = np.array([skew(x, axis=1) for x in X]).reshape(len(X), -1)
    kurtoses = np.array([kurtosis(x, axis=1) for x in X]).reshape(len(X), -1)
    covariances = np.array([np.cov(samples, rowvar=False)[0][1] for dist in X for samples in dist]).reshape(len(X), -1)
    return np.concatenate([means, stds, skews, kurtoses, covariances], axis=1)

def plot_feature_importance(model, feature_names, name):
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances.plot(kind='barh')
    plt.savefig('feature_imp/' + name + '_feat_importance.png')

def baseline(y_tr, y_ts):
    classification = isinstance(y_tr[0][0], str) or isinstance(y_tr[0][0], bool) or isinstance(y_tr[0][0], np.bool_)
    if len(y_tr[0]) > 1:
        raise NotImplemented("Baseline only ipmlemented for 1D outputs")
    y_tr = y_tr.flatten()
    y_ts = y_ts.flatten()
    if classification:
        majority_class = max(set(list(y_tr)), key=list(y_tr).count)
        train_score = sum(y_tr==majority_class)/len(y_tr)
        test_score = sum(y_ts==majority_class)/len(y_ts)
    else:
        mean = y_tr.mean()
        train_score = mean_squared_error([mean] * len(y_tr), y_tr)
        test_score = mean_squared_error([mean] * len(y_ts), y_ts)
    return train_score, test_score

def get_rdw(X):
    """
    X: [n_patients, n_dists, n_samples, 2]
    """
    means = np.array([np.mean(x[0][:, 0]) for x in X])
    stds = np.array([np.std(x[0][:, 0]) for x in X])
    rdws = stds/means*100
    rdws[np.where(np.isnan(rdws))] = 0
    return rdws.reshape(-1, 1)

def get_missing_indicator(X, imputation):
    if imputation == "zero":
        reduced_samples = (X == 0).all(axis=2)
    elif imputation == "nan":
        reduced_samples = (np.isnan(X)).all(axis=2)
    else:
        raise ValueError("Bad imputation value: ", imputation)
    reduced_features = (reduced_samples == True).all(axis=2)
    print("X", X.shape)
    print("reduced_samples", reduced_samples.shape)
    print("reduced_features", reduced_features.shape)
    return reduced_features  # [batch, n_dist]

def featurize_data(X_tr,  y_tr, X_ts, y_ts, name='', model='KNN', imputation='zero', missing_indicator=False, rdw='rdw'):
    if rdw == 'rdw':
        X_tr = get_rdw(X_tr)
        X_ts = get_rdw(X_ts)
    elif rdw == 'both':
        X_tr = np.hstack((get_moments(X_tr), get_rdw(X_tr)))
        X_ts = np.hstack((get_moments(X_ts), get_rdw(X_ts)))
    else:
        if missing_indicator:
            X_tr_mis = get_missing_indicator(X_tr, imputation)
            X_ts_mis = get_missing_indicator(X_ts, imputation)
        X_tr = get_moments(X_tr)
        X_ts = get_moments(X_ts)


    if imputation == "nan":
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(X_tr)
        X_tr = imp.transform(X_tr)
        X_ts = imp.transform(X_ts)
    elif imputation == "zero":
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        imp.fit(X_tr)
        X_tr = imp.transform(X_tr)
        X_ts = imp.transform(X_ts)
    if missing_indicator:
        assert len(X_tr) == len(X_tr_mis)
        assert len(X_ts) == len(X_ts_mis)
        X_tr = np.hstack([X_tr, X_tr_mis])
        X_ts = np.hstack([X_ts, X_ts_mis])
        assert len(X_tr) == len(X_tr_mis)
        assert len(X_ts) == len(X_ts_mis)
    return X_tr, X_ts, y_tr, y_ts

@profile
def train_sklearn_moments(X_tr,  y_tr, X_ts, y_ts, name='', model='KNN',
        imputation='zero', missing_indicator=False, rdw='rdw', featurized=False, id_file='', grid_search=False):
    if not featurized:
        X_tr, X_ts, y_tr, y_ts = featurize_data(X_tr,  y_tr, X_ts, y_ts, name='', model='KNN', imputation='zero', missing_indicator=False, rdw='rdw')
    classification = isinstance(y_tr[0][0], str) or isinstance(y_tr[0][0], bool) or isinstance(y_tr[0][0], np.bool_)
    if model=='KNN':
        parameters = {'n_neighbors': [3, 5, 9]}
        if classification:
            model = KNeighborsClassifier
        else:
            model = KNeighborsRegressor
    elif model=='RF':
        # parameters = {'n_estimators': [100, 200], 'min_samples_split': [2, 4, 8]}
        parameters = {'n_estimators': [100], 'min_samples_split': [30], 'n_jobs': [-1]}
        if classification:
            model = RandomForestClassifier
        else:
            model = RandomForestRegressor
    elif model=='GBC':
        parameters = {'n_estimators': [100, 200], 'learning_rate': [.001, .01, .1], 'min_samples_split': [2, 4, 8]}
        if classification:
            model = GradientBoostingClassifier
        else:
            model = GradientBoostingRegressor
    elif model=='RR':
        if classification:
            # parameters = {'C': [.001, .1, 1, 10, 100]}
            parameters = {'penalty': ['none']}
            model = LogisticRegression
        else:
            # parameters = {'alpha': [.001, .1, 1, 10, 100]}
            parameters = {'alpha': [1]}
            model = Ridge
    else:
        raise ArgumentError("Model not supported")
        
    #Double check there are no nans in input/output
    maskx = np.all(pd.notnull(X_tr), axis=1)
    # y is size (n, 1)
    masky = [i != "nan" for i in y_tr.flatten()]

    print(X_tr.shape)
    print(y_tr.shape)
    print(maskx, masky)
    X_tr = X_tr[maskx & masky]
    y_tr = y_tr[maskx & masky]
    print(X_tr.shape)
    print(y_tr.shape)
    print(maskx, masky)
    
    maskx = np.all(pd.notnull(X_ts), axis=1)
    masky = [i != "nan" for i in y_ts.flatten()]

    print(X_ts.shape)
    print(y_ts.shape)
    X_ts = X_ts[maskx & masky]
    y_ts = y_ts[maskx & masky]
    print(X_ts.shape)
    print(y_ts.shape)
    print('removed nans')
    assert len(X_tr) == len(y_tr)
    assert len(X_ts) == len(y_ts)
    if not os.path.exists(id_file + '_data.npz'):
        np.savez(id_file + '_data.npz', X_tr=X_tr, y_tr=y_tr, X_ts=X_ts, y_ts=y_ts)
    
    if grid_search:
        print('starting grid search')
        clf = GridSearchCV(model(), parameters)
        clf.fit(X_tr, y_tr)
        model = model(**clf.best_params_)
    else:
        model = model(**{k: v[0] for k, v in parameters.items()})
    print('model instantiated')
    model.fit(X_tr, y_tr)
    print('model fit')
    preds = model.predict(X_ts)
    print('test preds')
    pd.DataFrame.from_dict({'preds': preds.flatten(), 'labels': y_ts.flatten()}).to_csv(name + '.csv')
    # feature_names = ['mean0', 'mean1', 'std0', 'std1', 'skew0', 'skew1', 'kurtosis0', 'kurtosis1', 'cov']
    # if isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
#         plot_feature_importance(model, feature_names, name)
#     features = list(range(9)) #+ list(combinations(range(9), 2))
    # plot_partial_dependence(model, X_tr, features, feature_names, grid_resolution=20, percentiles=(0, 1))
    # plt.savefig('feature_imp/' + name + '_partial_dependence.png')
    jl.dump(model, f'model_{name}.jl')
    print('dumped model')
    np.savez(name + '.npz', X_tr=X_tr, y_tr=y_tr, X_ts=X_ts, y_ts=y_ts, preds=preds)
    print('saved preds')
    if classification:
        train_score = accuracy_score(y_tr, model.predict(X_tr))
        test_score = accuracy_score(y_ts, preds)
    else:
        train_score = mean_squared_error(y_tr, model.predict(X_tr))
        test_score = mean_squared_error(y_ts, preds)
    print('calculated scores')
    return train_score, test_score

