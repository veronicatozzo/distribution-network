python3.8 main.py -i rbc -o mean0 mean1 -m SetTransformer --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o mean0 mean1 -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o mean0 mean1 -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o mean0 mean1 -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01;

python3.8 main.py -i rbc -o std0 std1 -m SetTransformer --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o std0 std1 -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o std0 std1 -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o std0 std1 -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01;

python3.8 main.py -i rbc -o skew0 skew1 -m SetTransformer --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o skew0 skew1 -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o skew0 skew1 -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o skew0 skew1 -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01;

python3.8 main.py -i rbc -o median0 median1 -m SetTransformer --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o median0 median1 -m DeepSetsMean --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o median0 median1 -m DeepSetsMax --permute_subsamples --num_workers 0 --lr .01;
python3.8 main.py -i rbc -o median0 median1 -m DeepSetsSum --permute_subsamples --num_workers 0 --lr .01;