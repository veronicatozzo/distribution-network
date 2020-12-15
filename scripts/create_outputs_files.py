import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":

    data_folder = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/tables"

    files = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f))]

    #batch_length = 100000
    #i = 0
    #b = 0
    #while True:
    complete_table = []
    for i in range(len(files)):
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
            #i = i+1
        #if complete_table.shape[0]>=batch_length:
        #        break
        #if i >= len(files):
         #   break
        
    complete_table['file_id'] = complete_table.apply(lambda row : str(row['mrn'])+'_'+row['date'].split(' ')[0], axis = 1)

    complete_table['age'] = complete_table.apply(lambda row : ''.join([c for c in str(row['age']) if c.isdigit()]), axis=1)

    complete_table = complete_table[complete_table['age']!='']

        # create outputs

    table_ = complete_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
    table_.columns = ['mrn','date', 'Age', 'sex', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']

    # Age
    path = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/"
    table_['age'] = table_['Age']
    aux = table_[['mrn', 'date','folder', 'file_id', 'age','sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age']]
    aux.to_csv(path+'Age.csv')
    
#     path = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/"
#     table_['age'] = table_['Age']
#     aux = table_[['mrn', 'date','folder', 'file_id', 'age','sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age']]
#     aux['Age_binned'] = pd.cut(aux['Age'].values.astype(int), np.arange(0, 110, 10), right=True, labels=np.arange(0, 11))
#     aux.to_csv(path+'Age_binned.csv')
    
    # Sex
    table_ = complete_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
    table_.columns = ['mrn','date', 'age', 'Sex', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS',  'folder', 'file_id']
    aux = table_[['mrn', 'date',  'folder', 'file_id','age', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Sex']]
    aux.to_csv(path+'Sex.csv')

    
    # Age thresholded 
    table_ = complete_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
    complete_table['file_id'] = complete_table.apply(lambda row : str(row['mrn'])+'_'+row['date'].split(' ')[0], axis = 1)
    table_ = table_.astype({'age':int})
    table_['Age65'] = table_['age']>=65
    aux = table_[['mrn', 'date',  'folder', 'file_id','age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age65']]
    aux.to_csv(path+'Age65.csv')

    
    # Hematrocrit 
    table_ = complete_table[complete_table['HCT'].notnull()]
    table_ = table_[['mrn', 'date', 'age', 'sex', 'HCT', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
    table_.columns = ['mrn','date', 'age', 'sex','Hematocrit', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS','folder', 'file_id']
    aux = table_[['mrn', 'date', 'age','sex','folder', 'file_id','RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Hematocrit']]
    aux.to_csv(path+'Hematocrit.csv')

    aux = table_[['mrn', 'date', 'age','sex','folder', 'file_id','RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Hematocrit']]
    aux = aux[aux['sex'] =='F']
    aux.to_csv(path+'Hematocrit_female.csv')
    
    aux = table_[['mrn', 'date', 'age','sex','folder', 'file_id','RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Hematocrit']]
    aux = aux[aux['sex'] =='M']
    aux.to_csv(path+'Hematocrit_male.csv')
    

     #White blood count 
    table_ = complete_table.copy()
    table_['WBC'] = table_['WBC'].replace([np.inf, -np.inf], np.nan)
    table_ = table_[table_['WBC'].notnull()]
    table_ = table_[['mrn', 'date', 'age', 'sex', 'WBC', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS','folder', 'file_id']].copy()
    table_.columns = ['mrn','date','age', 'sex', 'WBC', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS','folder', 'file_id']
    aux = table_[['mrn', 'date','age', 'sex', 'folder', 'file_id', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'WBC']]
    aux.to_csv(path+'WBC.csv')
    print('im here')
    aux = table_[['mrn', 'date','age', 'sex', 'folder', 'file_id', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'WBC']]
    aux['WBC_binned'] = pd.cut(aux['WBC'], [0, 3, 11, 15, 30, 1000], right=True, labels=['0-3', '11-15', '15-30', '3-11', '30+'])
    aux.to_csv(path+'WBC_binned.csv')
#        b += 1
