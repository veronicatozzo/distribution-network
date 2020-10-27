import os
import h5py
import numpy as np
import torch
import inspect
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

from preprocessing import SetStandardScaler
from utils import shuffle_split_no_overlapping_patients, save_id_file
from utils import  save_id_file, read_id_file




# path_to_outputs = "/Users/vt908/Dropbox (Partners HealthCare)/Distribution-distribution regression/balanced_age/outputs"
# path_to_files = "/Users/vt908/Dropbox (Partners HealthCare)/Distribution-distribution regression/balanced_age"
# path_to_id_list = "/Users/vt908/Dropbox (Partners HealthCare)/Distribution-distribution regression/balanced_age/id_lists"
path_to_outputs = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/outputs"
path_to_files = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age"
path_to_id_list = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/id_files"


def select_one_patient_instance(ids_):
    """ Note: Will not be deterministic due to the set """
    ids = set()
    patient_ids = []
    for id_ in ids_:
        patient_id = id_.split('_')[0]
        if patient_id in patient_ids:
            continue
        ids.add(id_)
        patient_ids.append(patient_id)
    return ids

class FullSampleDataset(Dataset):
    """
        Params
        ------
        inputs: list of string 
            List of input distribution we want to take as input. 
            Provide options
        outputs: list of string
            List of outputs we want to predict. 
            Provide options
        dataset_name: string, 
            If provided uploads a pre-made dataset.
        normalization: string
            Options: 
                - 'sample': Standard normalization for each set of data 
                - 'all': Standard normalization across all input sets
                - estimator: any normalization applied to each single set. 
            In case dataset_name is provided normalization is ignored. 
    """
    def __init__(self, inputs=[], outputs = [],
                 id_file=None, num_subsamples=100, 
                 permute_subsamples=True, 
                 normalizer='all', test=False, stratify_by_patient=True):
        
        ids_ = set()
        self.outputs = []
        self.test = test
        for o in outputs:
            table_o = pd.read_csv(path_to_outputs+"/"+o+".csv", index_col=0)
            table_ids = set([str(table_o.iloc[i, 0])+'_'+table_o.iloc[i, 1].split(".")[0]
                        for i in range(table_o.shape[0])])
            ids_ = ids_.union(table_ids)
            if stratify_by_patient:
                ids_ = select_one_patient_instance(ids_)
            self.outputs.append(table_o)
       
        if (not os.path.exists(path_to_id_list+"/"+id_file)) and (id_file):
            aux = pd.DataFrame([[i.split('_')[0], i.split('_')[1]]
                                 for i in list(ids_)], columns = ['mrn', 'date'])
             #split train and test using shuffle split that cares for overlap 
            train, test = next(shuffle_split_no_overlapping_patients(aux, 
                                        train=0.8, n_splits=5))
            self.test_ids_ = np.array(list(ids_))[test]
            self.train_ids_ =  np.array(list(ids_))[train]
            save_id_file(self.train_ids_, self.test_ids_, id_file)
        else:
            id_list_train, id_list_test = read_id_file(id_file)
            self.test_ids_ = id_list_test
            self.train_ids_ =  id_list_train

        if self.test:
            self.ids_ = self.test_ids_
        else:
            self.ids_ = self.train_ids_
        
        if normalizer == 'all':
            self._setstandardscaler = []
            for input_type in inputs:
                ss = SetStandardScaler(stream=True)
                ss.fit([path_to_files+"/"+input_type+'/'+i+".npy" 
                        for i in self.train_ids_])
                self._setstandardscaler.append(ss)

        self.num_subsamples = num_subsamples
        self.permutate_subsamples =permute_subsamples
        self.normalizer = normalizer
        self.inputs = inputs
        
   
    def __getitem__(self, index):

        xs = []
        for i, input_type in enumerate(self.inputs):
            try:
                filename = path_to_files+"/"+input_type+'/'+self.ids_[index]+".npy" 
                x = np.load(filename)
                if self.normalizer == 'all':
                    x = self._setstandardscaler[i].transform([x])[0]
                elif isinstance(self.normalizer, TransformerMixin):
                    x = self.normalizer.fit_transform(x)
                else:
                    x = StandardScaler().fit_transform(x)

                if self.permutate_subsamples:
                    if x.shape[0] < self.num_subsamples: # corner case, added for soundness
                        perm = np.choice(np.arange(x.shape[0]), 
                                         self.num_subsamples, replace=True)
                    else:
                        perm = np.choice(np.arange(x.shape[0]),
                                         self.num_subsamples, replace=False)
                    xs.append(x[perm, :])
                else:
                    xs.append(x[:self.num_subsamples, :])
            except:
                xs.append(np.zeros((self.num_subsamples, 2)))

        ys = []
        mrn = self.ids_[index].split('_')[0]
        date = self.ids_[index].split('_')[1]
        for output in self.outputs:
            ys.append(output[(output['mrn']==int(mrn)) & (output['date'].str.contains(date))].iloc[0, -1])

        return np.array(xs), np.array(ys)

    def __len__(self):
        if self.test:
            return len(self.test_ids_)
        else:
            return len(self.train_ids_)
