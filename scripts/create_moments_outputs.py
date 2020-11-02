import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew

if __name__ == "__main__":
    output_dir = '/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/outputs'
    rbc_glob = glob.glob('/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/rbc/*.npy')
    moments = []
    for fname in rbc_glob:
        arr = np.load(fname)
        moments.append({
            'mrn': os.path.basename(fname).split('_')[0],
            'date': os.path.basename(fname).split('_')[1].split('npy')[0],
            # 'mean0': arr[:, 0].mean(),
            # 'std0': arr[:, 0].std(),
            # 'skew0': skew(arr[:, 0]),
            # 'median0': np.median(arr[:, 0]),
            # 'mean1': arr[:, 1].mean(),
            # 'std1': arr[:, 1].std(),
            # 'skew1': skew(arr[:, 1]),
            # 'median1': np.median(arr[:, 1]),
            '75quantile0': np.quantile(arr[:, 0], .75),
            '75quantile1': np.quantile(arr[:, 0], .75),
        })
    keys = moments[0].keys()
    df = pd.DataFrame(moments)
    for key in keys:
        if key not in ['mrn', 'date']:
            df[["mrn", "date", key]].to_csv(os.path.join(output_dir, key + ".csv"))


