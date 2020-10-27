python3.8 main.py -i rbc -o Hematocrit -m SetTransformer --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o Hematocrit -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o Hematocrit -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o Hematocrit -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01; 

python3.8 main.py -i rbc -o Age -m SetTransformer --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o Age -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o Age -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o Age -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01; 

python3.8 main.py -i rbc -o WhiteBloodCount -m SetTransformer --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o WhiteBloodCount -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o WhiteBloodCount -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01; 
python3.8 main.py -i rbc -o WhiteBloodCount -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01; 
