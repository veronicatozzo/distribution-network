import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class FullSampleDataset(Dataset):
    def __init__(self, test, start_on_gpu=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() and start_on_gpu else "cpu")
        fname='data.h5'
        with h5py.File(fname, 'r') as f:
            if test:
                train_data = np.array(f['test_cloud'])[:, ::100] # take 100 samples
                train_label = np.array(f['test_labels'])
            else:
                train_data = np.array(f['tr_cloud'])[:, ::100] # take 100 samples
                train_label = np.array(f['tr_labels'])
        train_label = train_label.reshape(train_label.shape[0], 1)
        self.train_data = torch.from_numpy(train_data).float().to(device)
        self.train_label = torch.from_numpy(train_label).float().to(device)

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]

    def __len__(self):
        return self.train_data.size(0)
