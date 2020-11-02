import os
import pickle
from functools import partial
import numpy as np
import pandas as pd
from scipy.stats import skew


def get_subsample_stat(fname, func, n):
    path_to_inputs = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/rbc"
    arr = np.load(os.path.join(path_to_inputs, fname))
    sample_idx = np.random.choice(range(len(arr)), n)
    sample = arr[sample_idx]
    return func(sample[:, 0]), func(sample[:, 1])

if __name__ == "__main__":
    for moment in ['mean', 'std', 'skew', 'median', '75quantile']:
        mse0s = []
        mse1s = []
        path_to_id_files = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/id_files"
        if moment == '75quantile':
            date = 'October-28-2020'
        else:
            date = 'October-27-2020'
        moment_pickle_file = '_'.join([moment+'0,'+moment+'1', 'rbc', date]) + '.txt'
        data = pickle.load(open(os.path.join(path_to_id_files, moment_pickle_file), 'rb'))
        func = None
        if moment == 'mean':
            func = np.mean
        elif moment == 'std':
            func = np.std
        elif moment == 'skew':
            func = skew
        elif moment == 'median':
            func = np.median
        elif moment == '75quantile':
            func = partial(np.quantile, q=.75)
        else:
            raise ValueError("Unsupported moment")
        
        test_files = data['test']
        labels0 = pd.read_csv('../balanced_age/outputs/' + moment + '0.csv')
        labels1 = pd.read_csv('../balanced_age/outputs/' + moment + '1.csv')
        for fname in test_files:
            subsample_stat0, subsample_stat1 = get_subsample_stat(fname, func, n=100)
            mrn, date = os.path.basename(fname).split('.txt')[0].split('_')
            label0 = labels0[(labels0['mrn']==int(mrn)) & (labels0['date'].str.contains(date))].iloc[0, -1]
            label1 = labels1[(labels1['mrn']==int(mrn)) & (labels1['date'].str.contains(date))].iloc[0, -1]
            mse0s.append((label0 - subsample_stat0)**2)
            mse1s.append((label1 - subsample_stat1)**2)
        print(
            moment,
            np.mean(mse0s), f"({np.std(mse0s)})",
            np.mean(mse1s), f"({np.std(mse1s)})"
        )
