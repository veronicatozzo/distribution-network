import os
import glob
from joblib import Parallel, delayed
import multiprocessing

path_to_data = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/data"

def write_moments(fname):
    dist = np.load(fname)
    moments = get_moments(dist)
    fname_out = fname.replace('/data/', '/moments/')
    os.makedirs(fname_out, exist_ok = True)
    np.save(fname_out, moments)

def get_moments(dist):
    """Arguments: dist [n_samples, 2]"""
    means = np.mean(dist, axis=0)
    stds = np.std(dist, axis=0)
    skews = skew(dist, axis=0)
    kurtoses = kurtosis(dist, axis=0)
    covariances = np.cov(samples, rowvar=False)[0][1]
    return np.concatenate([means, stds, skews, kurtoses, covariances], axis=1)

if __name__ == "__main__":
    data_files = glob.glob(path_to_data + '/**/*.npy', recursive=True)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(write_moments)(f) for f in data_files)
