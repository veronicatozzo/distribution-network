from sklearn.model_selection import ShuffleSplit
from itertools import combinations


def shuffle_split_no_overlapping_patients(data, train=0.8, n_splits=5):
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
                    diff = np.abs(data['date'].iloc[k] - data['date'].iloc[j])
                    if np.abs(data['date'].iloc[k] - data['date'].iloc[j]) <=180:
                        try:
                            allowed.remove(k)
                            allowed.remove(j)
                        except ValueError:
                            continue
            if len(allowed)>0:
                train_id += list(allowed)
            else:
                train_id.append(np.random.choice(np.where(data['mrn']==unique_id[i])[0], 1)[0])
        yield np.array(train_id), np.array(test_id)


def save_id_file(id_list, train, test, id_file):


def read_id_file(id_file, test=False):
    
