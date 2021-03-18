import datetime 

import pickle as pkl
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode 

from sklearn.model_selection import ShuffleSplit
from itertools import combinations
from collections import Counter
import collections


def wasserstein_2d(X1, X2):
    d = cdist(X1, X2)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(X1.shape[0], X2.shape[0])


def wasserstein_kernel(list_dist):
    kernel = np.zeros((len(list_dist), len(list_dist)))
    for i in range(len(list_dist)):
        for j in range(i, len(list_dist)):
            kernel[i,j] = wasserstein_2d(list_dist[i], list_dist[j])
    i_lower = np.tril_indices(len(list_dist), -1)
    kernel[i_lower] = kernel.T[i_lower]
    return kernel


def shuffle_split_no_overlapping_patients(data, train=0.8, n_splits=5, balance_age=False):
    unique_id = np.unique(data['mrn'])
    ss = ShuffleSplit(train_size=train, n_splits=n_splits)
    for train, test in ss.split(unique_id):
        test_id = [] 
        for i in test:
            test_id.append(np.random.choice(np.where(data['mrn']==unique_id[i])[0], 1)[0])
        train_id = []
        for i in train:
            ixs = np.where(data['mrn']==unique_id[i])[0]
            allowed = list(ixs)
            if not len(ixs)==1:
                for k, j in combinations(ixs, 2):
                    try:
                        diff = np.abs(datetime.datetime.strptime(data['date'].iloc[k], '%Y-%m-%d %H:%M:%S') - 
                                      datetime.datetime.strptime( data['date'].iloc[j], '%Y-%m-%d %H:%M:%S'))
                    except:
                        diff = np.abs(datetime.datetime.strptime(data['date'].iloc[k], '%Y-%m-%d') - 
                                      datetime.datetime.strptime( data['date'].iloc[j], '%Y-%m-%d'))
                    if diff.days <=180:
                        try:
                            allowed.remove(k)
                            allowed.remove(j)
                        except ValueError:
                            continue
            if len(allowed)>0:
                train_id += list(allowed)
            else:
                train_id.append(np.random.choice(np.where(data['mrn']==unique_id[i])[0], 1)[0])
        if balance_age:
            new_train_id = []
            ages = data['age'].iloc[train_id].values
            count_age = Counter(ages)
            count_age = collections.OrderedDict(sorted(count_age.items()))
            median = np.median([c for _, c in count_age.items()])
            for v, c  in count_age.items():
                ixs = np.where(ages==v)[0]
                if ixs.shape[0]> median:
                    new_ixs = np.random.choice(np.array(train_id)[ixs], int(median),
                                        replace=False)
                    new_train_id += list(new_ixs)
                else:
                    new_train_id += list(np.array(train_id)[ixs])
            train_id = new_train_id

            new_test_id = []
            ages = data['age'].iloc[test_id].values
            count_age = Counter(ages)
            median = np.median([c for _, c in count_age.items()])
            for v, c  in count_age.items():
                ixs = np.where(ages==v)[0]
                if ixs.shape[0]> median:
                    new_ixs = np.random.choice(np.array(test_id)[ixs], int(median),                         replace=False)
                    new_test_id += list(new_ixs)
                else:
                    new_test_id += list(np.array(test_id)[ixs])
            test_id = new_test_id
        yield np.array(train_id), np.array(test_id)


def save_id_file(train, test, id_file):
    to_save = dict(train=train,
                    test = test)
    with open(id_file, 'wb') as f:
        pkl.dump(to_save, f)

def read_id_file(id_file):
    with open(id_file, 'rb') as f:
        to_read = pkl.load(f)
    return to_read['train'], to_read['test']

def str_to_bool_arg(arg, name):
    """ wandb can't accept bool arguments in argparse, so we 
        use str arguments and translate into bool for downstream code use
    """
    if arg.lower() == 'true':
        value = True
    elif arg.lower() == 'false':
        value = False
    else:
        raise ValueError(f"Argument {name} is unaccepted value {arg}")
    return value