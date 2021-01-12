import os
import csv
import argparse
from datetime import datetime
#import wandb
import torch
from torch.utils.data import DataLoader

from deep_sets.models import SmallDeepSetMax, SmallDeepSetMean, SmallDeepSetSum
from set_transformer.models import SmallSetTransformer
from src.dataset import FullSampleDataset, FullLargeDataset
from src.train import train_nn as train


# os.environ["WANDB_API_KEY"] = "ec22fec7bdd7579e0c42b8d29465922af4340148"  # "893130108141453e3e50e00010d3e3fced11c1e8"

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('-i', '--inputs', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the distribution inputs')
parser.add_argument('-o', '--outputs', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the prediction outputs')
parser.add_argument('-m', '--model', type=str,
                    help='string name for model type')

parser.add_argument('--id_file', type=str, default='', help='filename of the ids to use')
parser.add_argument('--data_large', dest='data_large', action='store_true')
parser.add_argument('--num_samples', type=int, default=-1,
                    help='number of samples to use in each distribution. note that non-neural models use all subsamples if -1')
parser.add_argument('--num_subsamples', type=int, default=100,
                    help='number of samples to use in each distribution. note that non-neural models use all subsamples if -1')
parser.add_argument('--permute_subsamples', dest='permute_subsamples', action='store_true')
parser.add_argument('--normalizer', type=str, help='name of the normalizer', default='none')
parser.add_argument('--imputation', type=str, help='name of the normalizer', default='zero')
parser.add_argument('--rdw', type=str,  default='none')
parser.add_argument('--missing_indicator', dest='missing_indicator', action='store_true', help='whether to add missing indicators for each dist')


parser.add_argument('--output_file', type=str, help='name of the normalizer', default='baselines.csv')


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu workers in the data loader')

parser.add_argument('--n_enc_layers', type=int, default=2)
parser.add_argument('--n_dec_layers', type=int, default=2)
parser.add_argument('--n_hidden_units', type=int, default=64)

parser.add_argument('--lr', type=float, default=.01)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--gamma', type=int, default=.1)

parser.add_argument('--name', type=str, help='name of the experiment in wandb',
                    default='')
parser.add_argument('--save', dest='save', action='store_true', help='whether to save the model')
parser.add_argument('--local_testing', dest='local_testing', action='store_true', help='flag to signify local testing')
args = parser.parse_args()

model_dict = {
    "SetTransformer":SmallSetTransformer,
    "DeepSetsMax":SmallDeepSetMax,
    "DeepSetsMean": SmallDeepSetMean,
    "DeepSetsSum": SmallDeepSetSum,
}

outputs_dict = dict(
    Age={"type": "regression"},
    Hematocrit={"type": "regression"},
    Hematocrit_female={"type": "regression"},
    Hematocrit_male={"type": "regression"},
    Ferritin={"type": "regression"},
    Ferritin_female={"type": "regression"},
    Ferritin_male={"type": "regression"},
    Cholesterol={"type": "regression"},
    A1c={"type": "regression"},
    Sex={"type": "regression"},
    Age65={"type": "classification", "num_classes": 2},
)
print(torch.cuda.is_available())

if __name__ == "__main__":
    print(args.outputs)
    print(args.inputs)
    
    if args.name:
        name = args.name
    else:
        name = '_'.join([args.model, ','.join(args.outputs), ','.join(args.inputs)])
        #name = '_'.join([args.model, 'full', 'rbc'])
    # wandb.init(project="distribution-regression", name=name)
   # wandb.init(project="blood-distribution-moments", name=name)
    #wandb.config.update(args, allow_val_change=True)

    if args.id_file:
        id_file = args.id_file
    else:
        if args.data_large:
            path_to_id_files = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/data_large/id_files"
        else:
            path_to_id_files = "/misc/vlgscratch5/RanganathGroup/lily/blood_dist/balanced_age/id_files"
        id_file = os.path.join(path_to_id_files, '_'.join([','.join(args.outputs), ','.join(args.inputs), str("{:%B-%d-%Y}.txt".format(datetime.now()))]))

    data_config = {
        'inputs': args.inputs,
        'outputs': args.outputs,
        'id_file': id_file,
        'num_subsamples': args.num_subsamples,
        'permute_subsamples': args.permute_subsamples,
        'normalizer': args.normalizer,
    }

    if args.data_large:
        Dataset = FullLargeDataset
    else:
        Dataset = FullSampleDataset
       
    train_data = Dataset(test=False, **data_config)
    test_data = Dataset(test=True, **data_config)
    train_generator = DataLoader(train_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=True)
    test_generator = DataLoader(test_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=True)
    
    model_params = {
        'n_outputs': len(args.outputs),
        'n_inputs': len(args.inputs),
        'n_enc_layers': args.n_enc_layers,
        'n_dec_layers': args.n_dec_layers,
        'n_hidden_units': args.n_hidden_units,
    }
    model = model_dict[args.model](**model_params)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)
    model, train_score, test_score, train_losses, test_losses = train(model, name, optimizer, scheduler, train_generator,
                                                                      test_generator, 
                                                                      classification=outputs_dict[args.outputs[0]]['type']=='classification',
                                                                     n_epochs=args.n_epochs)
    
        
    print(train_score, test_score)
    with open(args.output_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([','.join(args.inputs), ','.join(args.outputs), args.model, args.imputation, args.missing_indicator, args.normalizer, train_score, test_score])
    if args.save:
        torch.save(model, args.name + '.pt')
        np.save(train_losses, args.name+'train_losses.npy')
        np.save(train_losses, args.name+'test_losses.npy')