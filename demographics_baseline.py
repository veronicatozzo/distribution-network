import os
import csv
import argparse
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from utils import read_id_file

parser = argparse.ArgumentParser(description='Results summary')
parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('-o', '--outputs', type=str, default='', help='str output')
parser.add_argument('-m', '--model', type=str, default='', help='filename of the ids to use')
parser.add_argument('--id_file', type=str, default='', help='filename of the ids to use')
parser.add_argument('--imputation', type=str, help='name of the normalizer', default='zero')
parser.add_argument('--output_file', type=str, help='name of the normalizer', default='baselines.csv')
args = parser.parse_args()

path_to_outputs = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs"

if __name__ == "__main__":
    # NOTE: length of outputs is 1
    data = pd.read_csv(os.path.join(path_to_outputs, args.outputs + '.csv'), index_col=0)
#     data['Sex'] = 'M'
#     data['Age'] = 50
    data['Sex'] = data['sex'] == 'M'
    data['Sex_nan'] = pd.isnull(data.Sex)
    data['Age_nan'] = pd.isnull(data.age)
    data['date'] = data['date'].apply(lambda s: s.split('.')[0])
    id_list_train, id_list_test = read_id_file(args.id_file)
    mrns_train = list(map(lambda s: s.split('_')[0], id_list_train))
    mrns_test = list(map(lambda s: s.split('_')[0], id_list_test))
    dates_train = list(map(lambda s: s.split('_')[1], id_list_train))
    dates_test = list(map(lambda s: s.split('_')[1], id_list_test))
    data['date'] = list(map(lambda  s: s.split(' ')[0], data['date'].astype(str).values))
    train_mask = data['mrn'].astype(str).isin(mrns_train)&data['date'].astype(str).isin(dates_train)
    print(np.any(train_mask))
    if args.imputation == 'zero':
        data[['age', 'Sex']].fillna(0, inplace=True)
    elif args.imputation == 'nan':
        data[['age']].fillna(data[train_mask].Age.mean(), inplace=True)
        data[['Sex']].fillna(data[train_mask].Sex.mode(), inplace=True)
    train = data[train_mask]
    test = data[data['mrn'].astype(str).isin(mrns_test)&data['date'].astype(str).isin(dates_test)]
    print(train.shape)
    X_tr = train[['age', 'Sex', 'Age_nan', 'Sex_nan']]
    y_tr = train.iloc[:, -1]
    X_ts = test[['age', 'Sex', 'Age_nan', 'Sex_nan']]
    y_ts = test.iloc[:, -1]
    print(y_tr.values.shape)
    classification = isinstance(y_tr.values[0], str) or isinstance(y_tr.values[0], bool) or isinstance(y_tr.values[0], np.bool_)
    if args.model=='KNN':
        parameters = {'n_neighbors': [3, 5, 9]}
        if classification:
            model = KNeighborsClassifier
        else:
            model = KNeighborsRegressor
    elif args.model=='RF':
        parameters = {'n_estimators': [100, 200], 'min_samples_split': [2, 4, 8]}
        if classification:
            model = RandomForestClassifier
        else:
            model = RandomForestRegressor
    elif args.model=='GBC':
        parameters = {'n_estimators': [100, 200], 'learning_rate': [.001, .01, .1], 'min_samples_split': [2, 4, 8]}
        if classification:
            model = GradientBoostingClassifier
        else:
            model = GradientBoostingRegressor
    elif args.model=='RR':
        if classification:
            parameters = {'C': [.001, .1, 1, 10, 100]}
            model = LogisticRegression
        else:
            parameters = {'alpha': [.001, .1, 1, 10, 100]}
            model = Ridge
    else:
        raise ValueError("Model not supported")
    clf = GridSearchCV(model(), parameters)
    clf.fit(X_tr, y_tr)
    model = model(**clf.best_params_)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_ts)
    name = '_'.join([args.model, ','.join([args.outputs]), ','.join(['demo'])])
    np.savez(name + '.npz', X_tr=X_tr, y_tr=y_tr, X_ts=X_ts, y_ts=y_ts, preds=preds)
    pd.DataFrame.from_dict({'preds': preds, 'labels': y_ts}).to_csv(name + '.csv')
    normalizer = 'all'
    missing_indicator = True
    if classification:
        train_score = accuracy_score(y_tr, model.predict(X_tr))
        test_score = accuracy_score(y_ts, preds)
    else:
        train_score = mean_squared_error(y_tr, model.predict(X_tr))
        test_score = mean_squared_error(y_ts, model.predict(X_ts))
    with open(args.output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([','.join(['demo']), ','.join([args.outputs]), args.model, args.imputation, missing_indicator, normalizer, train_score, test_score])
