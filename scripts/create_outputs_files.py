import numpy as np
import pandas as pd

from datetime import datetime 
from os import listdir
from os.path import isfile, join

def find_matching_distributions(data, output):
    data_folder = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/tables"
    final_output_table = []
    
    for year in range(2006, 2013):
        for month in range(1,12):
            if month < 10:
                date = '0'+str(month)
            else:
                date = str(month)
            # select data for that month
            data_month = data[data['date'].str.startswith(date)]

            # select all tables of that month
            files = [join(data_folder, f) for f in listdir(data_folder) 
                     if isfile(join(data_folder, f)) and str(f).startswith(str(year)+'.'+date)]
            if len(files) == 0:
                continue
            complete_table = []
            for i, f in enumerate(files):
                t = pd.read_csv(files[i], header=None)
                t.columns = ['mrn','date','age','sex', 'RBC','RETICS','BASOS','PAROX','PLTS','MCV','MCH','PCV','MPC','HCT','WBC']
                splits = files[i].split('/')[-1].split('_')
                fold, subfold = splits[0], splits[1].split('.')[0]
                t['folder'] = '/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/data/'+fold+'/'+subfold+'/'
                if isinstance(complete_table, list):
                    complete_table = t
                else:
                    complete_table = complete_table.append(t)

            #for each row of data of that month search for correspondance in the distributions
            for j in range(data_month.shape[0]):
                date = datetime.strptime(data_month.iloc[j, 1], '%m/%d/%Y %H:%M') 
                aux = complete_table[complete_table.iloc[:, 0]==data_month.iloc[j, 0]]
                distances = np.array([(datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') - date).days 
                                      for d in aux.iloc[:, 1]]).astype(np.int16)
                ixs = np.where(distances == 0)
                if ixs[0].size == 0:
                    ixs = np.where(np.logical_or(distances == 1,  distances ==-1))
                if ixs[0].size == 0:
                    continue
                ix = ixs[0][0]

                if len(final_output_table) == 0:
                    final_output_table = pd.DataFrame(np.hstack((np.array(aux.iloc[ix, :]), 
                                                                 data_month.iloc[j, 3])).reshape(1, aux.shape[1]+1),
                                  columns = list(aux.columns)+[output])
                else:
                    final_output_table = final_output_table.append(
                        pd.DataFrame(np.hstack((np.array(aux.iloc[ix, :]), 
                                                data_month.iloc[j, 3])).reshape(1, aux.shape[1]+1),
                                  columns = list(aux.columns)+[output]))
    final_output_table['file_id'] = final_output_table.apply(lambda row : 
                                                             str(row['mrn'])+'_'+row['date'].split(' ')[0], axis = 1)

    final_output_table['age'] = final_output_table.apply(lambda row : 
                                                         ''.join([c for c in str(row['age']) if c.isdigit()]), axis=1)

    final_output_table = final_output_table[final_output_table['age']!='']

    table_ = final_output_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 
                                 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id', output]].copy()
    table_ = table_[table_[output].notnull()]
    path = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/"
    output_ = output[0].upper()+output[1:]
    table_.to_csv(path+output_+'.csv')
    
    if output.lower() == 'ferritin':
        table_ = table_[table_.iloc[:, -1]<=40]
        table_.to_csv(path+'Ferritin40.csv')
        
        table_ = final_output_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS',
                                     'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id', 'Ferritin']].copy()
        table_ = table_[(table_['sex'] == 'F') & (table_.iloc[:, -1] <= 10)]
        table_.to_csv(path+'Ferritin_female.csv')
        
        table_ = final_output_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 
                                     'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id', 'Ferritin']].copy()
        table_ = table_[(table_['sex'] == 'M') & (table_.iloc[:, -1] <= 30)]
        table_.to_csv(path+'Ferritin_male.csv')



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
        t['folder'] = '/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/data/'+fold+'/'+subfold+'/'
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
    table_ = complete_table[complete_table['Age'].notnull()]
    aux = table_[['mrn', 'date','folder', 'file_id', 'age','sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age']]
    aux.to_csv(path+'Age.csv')
    
    path = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/"
    table_['age'] = table_['Age']
    table_ = complete_table[complete_table['Age'].notnull()]
    aux = table_[['mrn', 'date','folder', 'file_id', 'age','sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Age']]
    aux['Age_binned'] = pd.cut(aux['Age'].values.astype(int), np.arange(0, 110, 10), right=False, labels=np.arange(0, 10))
    aux.to_csv(path+'Age_binned.csv')
    
    # Sex
    table_ = complete_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
    table_.columns = ['mrn','date', 'age', 'Sex', 'RBC','RETICS', 'BASOS', 'PAROX', 'PLTS',  'folder', 'file_id']
    table_ = complete_table[complete_table['Sex'].notnull()]
    aux = table_[['mrn', 'date',  'folder', 'file_id','age', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'Sex']]
    aux.to_csv(path+'Sex.csv')

    
    # Age thresholded 
    table_ = complete_table[['mrn', 'date', 'age', 'sex', 'RBC', 'RETICS', 'BASOS', 'PAROX', 'PLTS', 'folder', 'file_id']].copy()
    complete_table['file_id'] = complete_table.apply(lambda row : str(row['mrn'])+'_'+row['date'].split(' ')[0], axis = 1)
    table_ = table_.astype({'age':int})
    table_['Age65'] = table_['age']>=65
    table_ = complete_table[complete_table['Age65'].notnull()]
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



    #------------Ferritin
    path = '/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/query_labs/'
    tests = pd.read_csv(path+'ferritin.csv')
    tests = tests.append(pd.read_csv(path+'Ferritin_2007_2012.csv'))
    tests = tests[tests['Test']=='Ferritin']
    tests.sort_values(['date'], axis=0, inplace=True)
    tests = tests[tests['date']!='NaT']
    find_matching_distributions(tests, 'Ferritin')
    
    
    #------------Cholesterol and A1c 
    path = '/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/query_labs/cholesterol_a1c/'
    files = [join(path, f) for f in listdir(path)]
    for i, f in enumerate(files):
        if i == 0:
            tests = pd.read_csv(f)
        else:
            tests = tests.append(pd.read_csv(f))
    tests.sort_values(['date'], axis=0, inplace=True)
    tests = tests[tests['date']!='NaT']
    A1c = tests[tests['Test'].str.contains('A1C')]
    find_matching_distributions(A1c, 'A1c')
    cholesterol = tests[tests['Test'].str.contains('Cholesterol')]
    find_matching_distributions(cholesterol, 'Cholesterol')
