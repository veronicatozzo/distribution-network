import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":

    data_folder = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/tables"

    files = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f))]

    files

    batch_length = 100000
    i = 0
    b = 0
    while True:
        complete_table = []
        while i<len(files):
            t = pd.read_csv(files[i], header=None)
            t.columns = ['mrn','date','age','sex', 'RBC','RETICS','BASOS','PAROX','PLTS','MCV','MCH','PCV','MPC','HCT','WBC']
            splits = files[i].split('/')[-1].split('_')
            fold, subfold = splits[0], splits[1].split('.')[0]
            t['folder'] = '/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/'+fold+'/'+subfold+'/'
            if isinstance(complete_table, list):
                complete_table = t
            else:
                complete_table = complete_table.append(t)
            #print(complete_table)
            i = i+1
            if complete_table.shape[0]>=batch_length:
                break
        if i >= len(files):
            break
        
        complete_table['file_id'] = complete_table.apply(lambda row : str(row['mrn'])+'_'+row['date'].split(' ')[0], axis = 1)

        complete_table['age'] = complete_table.apply(lambda row : ''.join([c for c in str(row['age']) if c.isdigit()]), axis=1)

        complete_table = complete_table[complete_table['age']!='']

        # create outputs

        table_ = complete_table[['mrn', 'date', 'age', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
        table_.columns = ['mrn','date', 'Age', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']

        # Age
        path = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/"
        table_['age'] = table_['Age']
        aux = table_[['mrn', 'date','folder', 'file_id', 'age', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age']]
        aux.to_csv(path+'Age'+str(b)+'.csv')

        # Sex
        table_ = complete_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
        table_.columns = ['mrn','date', 'age', 'Sex', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS',  'folder', 'file_id']
        aux = table_[['mrn', 'date',  'folder', 'file_id','age', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Sex']]
        aux.to_csv(path+'Sex'+str(b)+'.csv')

        # Age thresholded 
        table_ = complete_table[['mrn', 'date', 'age', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
        complete_table['file_id'] = complete_table.apply(lambda row : str(row['mrn'])+'_'+row['date'].split(' ')[0], axis = 1)
        table_ = table_.astype({'age':int})
        table_['Age65'] = table_['age']>=65
        aux = table_[['mrn', 'date',  'folder', 'file_id','age','RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age65']]
        aux.to_csv(path+'Age65'+str(b)+'.csv')

        # Hematrocrit 
        table_ = complete_table[complete_table['HCT'].notnull()]
        table_ = table_[['mrn', 'date', 'age', 'sex', 'HCT', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
        table_.columns = ['mrn','date', 'age', 'sex','Hematocrit', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS','folder', 'file_id']
        aux = table_[['mrn', 'date', 'age','sex','folder', 'file_id','RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Hematocrit']]
        aux.to_csv(path+'Hematocrit'+str(b)+'.csv')

        #White blood count 
        table_ = complete_table.copy()
        table_['WBC'] = table_['WBC'].replace([np.inf, -np.inf], np.nan)
        table_ = table_[table_['WBC'].notnull()]
        table_ = table_[['mrn', 'date', 'age', 'WBC', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS','folder', 'file_id']].copy()
        table_.columns = ['mrn','date','age', 'WBC', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS','folder', 'file_id']
        aux = table_[['mrn', 'date','age', 'folder', 'file_id', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'WBC']]
        aux.to_csv(path+'WBC'+str(b)+'.csv')
        b += 1