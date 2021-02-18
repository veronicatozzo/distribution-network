from sklearn.preprocessing import StandardScaler
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
from tqdm import tqdm 

data_folder = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/tables"


f= open("/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/list_of_moments.txt","w+")
f.close()
print('----------------------starting---------------------')    
for year in range(2006, 2007):
       
    print('----------------------'+str(year)+'---------------------')
    for month in tqdm(range(1,12)):
        complete_table = []
        if month < 10:
            date = '0'+str(month)
        else:
            date = str(month)
        
        # select all tables of that month
        files = [join(data_folder, f) for f in os.listdir(data_folder) 
                 if isfile(join(data_folder, f)) and str(f).startswith(str(year)+'.'+date)]
        if len(files) == 0:
            continue
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
                
        complete_table = complete_table[['mrn','date','age','sex','RBC','RETICS','BASOS','PAROX','PLTS','folder']]

        aux = pd.DataFrame(np.zeros((complete_table.shape[0], 34)))

        for type_ in ['RBC', 'BASOS', 'PEROX', 'RETIC']:
            for i in range(complete_table.shape[0]):
                file_id = str(complete_table.iloc[i, 0])+'_'+ complete_table.iloc[i, 1].split(' ')[0]+'_'+type_+'.csv'
                path = complete_table.iloc[0, -1]
                path = path[0:60] +'moments'+path[64:]
                try:
                    moments = pd.read_csv(path+file_id, index_col = 0)
                    aux.iloc[i,:] = moments.iloc[0,:]
                    aux.columns = moments.columns 
                except:
                    aux.iloc[i, :] = np.array([np.nan]*34).reshape(1, 34)


            for c in aux.columns:
                out = pd.concat([complete_table, aux[c]], axis=1)#(aux[c]-np.nanmean(aux[c]))/np.nanstd(aux[c])], axis=1)
                if isfile('/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/'+c+'_'+type_+'.csv'):
                    out.to_csv('/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/'+c+'_'+type_+'.csv', 
                               mode='a', header=False)
                else:
                    out.to_csv('/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/'+c+'_'+type_+'.csv')
                print('/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/outputs/'+c+'_'+type_)
            with open("/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/list_of_moments.txt","a+") as f:
                f.write(c+'_'+type_+"\n")
        

