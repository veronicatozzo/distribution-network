import os
import glob
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import traceback
# from concurrent.futures import ProcessPoolExecutor

path_to_data = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/data"

def write_moments(fname):
    try:
        fname_out = fname.replace('/data/', '/moments/').replace('.npy', '.csv')
        if os.path.exists(fname_out):
            continue
        else:
            dist = np.load(fname)
            moments = get_moments(dist)
            os.makedirs(os.path.dirname(fname_out), exist_ok = True)
            moments.to_csv(fname_out)
        # np.save(fname_out, moments)
    except:
        log_path = os.path.join(os.path.dirname(path_to_data), 'moments_err_log.txt'
        err = traceback.format_exc()
        with open(log_path, 'w') as f:
            f.write(fname + '\n')
            f.write(err)
            f.write('\n\n')

def get_rdw(X):
    """
    X: [n_samples, 2]
    """
    mean = np.mean(X[:, 0])
    std = np.std(X[:, 0])
    rdw = std/mean * 100
    return rdw.reshape(-1)

def get_moments(dist):
    """Arguments: dist [n_samples, 2]"""
    means = np.mean(dist, axis=0)
    stds = np.std(dist, axis=0)
    skews = skew(dist, axis=0)
    kurtoses = kurtosis(dist, axis=0)
    covariance = np.cov(dist, rowvar=False)[0][1].reshape(-1)
    moments = [means, stds, skews, kurtoses]
    for quantile in list(np.arange(.1, 1, .1)) + [.25, .5, .75]:
        moments.append(np.quantile(dist, quantile, axis=0))
    moments.append(get_rdw(dist))
    moments.append(covariance)
    df = pd.DataFrame([np.concatenate(moments, axis=0)])
    cols = (
        ['mean', 'std', 'skew', 'kurtosis'] + 
        ['quantile' + str(round(q, 1)) for q in list(np.arange(.1, 1, .1))] + 
        ['quantile' + str(q) for q in [.25, .5, .75]])
    cols = [col + '_' + str(i) for col in cols for i in range(2)]
    df.columns = cols + ['rdw', 'cov']
    return df

if __name__ == "__main__":
    data_files = glob.glob(path_to_data + '/**/*.npy', recursive=True)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(write_moments)(f) for f in data_files)