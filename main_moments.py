import os
import csv
import sys
import argparse
import glob
from datetime import datetime
import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils import read_id_file

from src.train import train_sklearn_moments, baseline

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('-i', '--inputs', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the distribution inputs')
parser.add_argument('-o', '--outputs', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the prediction outputs')
parser.add_argument('-m', '--model', type=str,
                    help='string name for model type')

parser.add_argument('--id_file', type=str, default='', help='filename of the ids to use')
parser.add_argument('--normalizer', type=str, help='name of the normalizer', default='all')
parser.add_argument('--imputation', type=str, help='name of the normalizer', default='zero')
parser.add_argument('--rdw', type=str,  default='none')
parser.add_argument('--missing_indicator', dest='missing_indicator', action='store_true', help='whether to add missing indicators for each dist')
parser.add_argument('--output_file', type=str, help='name of the normalizer', default='baselines.csv')

parser.add_argument('--name', type=str, help='name of the experiment in wandb',
                    default='')

args = parser.parse_args()

path_to_data = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/moments"
path_to_outputs="/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs"
cell_types = ['RBC', 'RETIC', 'PLT', 'BASOS', 'PEROX']

def get_data(id_list, imputation, missing_indicator, rdw):
    all_data = []
    is_nan = {}
    for id_name in id_list:
        ## remove timestamp (only for the bad Dec 15 id_files)
        # id_name = id_name.split(' ')[0]
        date = id_name.split('_')[1]
        year, month, date = date.split('-')
        feats = []
        for cell_type in cell_types:
            path = os.path.join(path_to_data, '.'.join([year, month]), '-'.join([month, date]) + '*', cell_type, id_name + '_' + cell_type + '*.csv')
            fname = glob.glob(path)
            print(fname)
            if len(fname) == 0:
                print('doesnt exist')
                is_nan[cell_type + '_isnan'] = [True]
                cols = (
                    ['mean', 'std', 'skew', 'kurtosis'] + 
                    ['quantile' + str(round(q, 1)) for q in list(np.arange(.1, 1, .1))] + 
                    ['quantile' + str(q) for q in [.25, .75]])
                cols = [col + '_' + str(i) for col in cols for i in range(2)]
                cols = cols + ['rdw', 'cov']
                if imputation == 'zero':
                    imputed_value = 0
                else:
                    imputed_value = np.nan
                df = pd.DataFrame([[imputed_value] * len(cols)], columns=cols)
            else:
                is_nan[cell_type + '_isnan'] = [False]
                # assert len(fname) == 1
                fname = fname[0]
                df = pd.read_csv(fname, index_col=0)
                df.drop(['quantile0.5_0.1', 'quantile0.5_1.1'], axis=1, inplace=True)
            df.columns = [cell_type + '_' + c for c in df.columns]
            if cell_type != 'RBC':
                df.drop(cell_type + '_' + 'rdw', axis=1, inplace=True)
            else:
                if rdw == 'rdw':
                    df = df[[cell_type + '_' + 'rdw']]
                elif rdw == 'both':
                    pass
                else:
                    df.drop(cell_type + '_' + 'rdw', axis=1, inplace=True)
            print(cell_type,  df.shape)
            if cell_type == 'RETIC':
                print(df.columns)
            feats.append(df)
        all_feats = pd.concat(feats, axis=1)
        all_feats['file_id'] = id_name
        if missing_indicator:
            dist_isnan = pd.DataFrame.from_dict(is_nan)
            all_feats = pd.concat([all_feats, dist_isnan], axis=1)
        all_data.append(all_feats)
        print([el.shape for el in all_data])
        print([set(el.columns) - set(all_data[0].columns) for el in all_data])
    return pd.concat(all_data, axis=0)


if __name__ == "__main__":
    if args.name:
        name = args.name
    else:
        name = '_'.join([args.model, ','.join(args.outputs), ','.join(args.inputs)])

    if args.id_file:
        id_file = args.id_file
    else:
        path_to_id_files = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/id_files"
        id_file = os.path.join(path_to_id_files, '_'.join([','.join(args.outputs), ','.join(args.inputs), str("{:%B-%d-%Y}.txt".format(datetime.now()))]))

    data_config = {
        'inputs': args.inputs,
        'outputs': args.outputs,
        'id_file': id_file,
        'normalizer': args.normalizer,
        'imputation': args.imputation
    }

    # TODO: maybe parallelize
    if not os.path.exists(id_file):
        raise ValueError("Id file has not been created: {id_file}")
    if len(args.outputs) > 1:
        raise NotImplementedError("Only single-output prediction is supported")
    id_list_train, id_list_test = read_id_file(id_file)
    output = args.outputs[0]
    table_o = pd.read_csv(path_to_outputs+"/"+output+".csv", index_col=0)
    table_ids = set(table_o['file_id'].values)
    print(len(id_list_train), len(id_list_test))
    id_list_train = list(set(id_list_train).intersection(table_ids))
    id_list_test = list(set(id_list_test).intersection(table_ids))
    print(len(id_list_train), len(id_list_test))
    if args.model == 'baseline':
        y_tr = table_o[table_o.file_id.isin(id_list_train)][output]
        y_ts = table_o[table_o.file_id.isin(id_list_test)][output]
        train_score, test_score = baseline(y_tr, y_ts)
    else:
        X_tr = get_data(id_list_train, args.imputation, args.missing_indicator, args.rdw)
        X_ts = get_data(id_list_test, args.imputation, args.missing_indicator, args.rdw)
        y_tr = table_o[table_o.file_id.isin(id_list_train)][[output, 'file_id']]
        y_ts = table_o[table_o.file_id.isin(id_list_test)][[output, 'file_id']]
        y_tr.drop_duplicates('file_id', inplace=True)
        y_ts.drop_duplicates('file_id', inplace=True)
        # join so everything is in order
        tr = X_tr.merge(y_tr, on='file_id', how='left')
        ts = X_ts.merge(y_ts, on='file_id', how='left')
        print(len(tr), len(X_tr), len(y_tr))
        print(len(ts), len(X_ts), len(y_ts))
        assert len(tr) == len(X_tr)
        assert len(ts) == len(X_ts)
        y_tr = tr[output]
        X_tr = tr.drop([output, 'file_id'], axis=1)
        y_ts = ts[output]
        X_ts = ts.drop([output, 'file_id'], axis=1)
        if args.model == 'RR':
            scaler = StandardScaler()
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_ts = scaler.transform(X_ts)
        if args.imputation == 'nan':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            imp.fit(X_tr)
            X_tr = imp.transform(X_tr)
            X_ts = imp.transform(X_ts)
        y_tr = table_o[table_o.file_id.isin(id_list_train)][[output, 'file_id']]
        y_ts = table_o[table_o.file_id.isin(id_list_test)][[output, 'file_id']]
        print(X_tr[0])
        train_score, test_score = train_sklearn_moments(X_tr, y_tr.values.reshape((-1, 1)), X_ts, y_ts.values.reshape((-1, 1)), name=name, model=args.model, id_file=id_file, featurized=True)
    with open(args.output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([','.join(args.inputs), ','.join(args.outputs), args.model, args.imputation, args.missing_indicator, args.normalizer, train_score, test_score])
