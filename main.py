import os
import argparse
from datetime import datetime
import wandb
import torch
from torch.utils.data import DataLoader

from deep_sets.models import SmallDeepSetMax, SmallDeepSetMean, SmallDeepSetSum
from set_transformer.models import SmallSetTransformer
from src.dataset import FullSampleDataset
from src.train import train


os.environ["WANDB_API_KEY"] = "ec22fec7bdd7579e0c42b8d29465922af4340148"  # "893130108141453e3e50e00010d3e3fced11c1e8"

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('-i', '--inputs', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the distribution inputs')
parser.add_argument('-o', '--outputs', metavar='N', type=str, nargs='+',
                    help='a list of strings denoting the prediction outputs')
parser.add_argument('-m', '--model', type=str,
                    help='string name for model type')

parser.add_argument('--id_file', type=str, default='', help='filename of the ids to use')
parser.add_argument('--num_subsamples', type=int, default=100, help='number of samples to use in each distribution')
parser.add_argument('--permute_subsamples', dest='permute_subsamples', action='store_true')
parser.add_argument('--normalizer', type=str, help='name of the normalizer')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu workers in the data loader')

parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--step_size', type=int, default=50)
parser.add_argument('--gamma', type=int, default=.1)

parser.add_argument('--name', type=str, help='name of the experiment in wandb',
                    default='')
parser.add_argument('--save', dest='save', action='store_true', help='whether to save the model')
args = parser.parse_args()

model_dict = dict(
    SetTransformer=SmallSetTransformer,
    DeepSetsMax=SmallDeepSetMax,
    DeepSetsMean=SmallDeepSetMean,
    DeepSetsSum=SmallDeepSetSum,
)

outputs_dict = dict(
    Age={"type": "regression"},
    Hematocrit={"type": "regression"},
    Age70={"type": "classification", "num_classes": 2},
)
print(torch.cuda.is_available())

if __name__ == "__main__":
    if args.name:
        name = args.name
    else:
        name = '_'.join([args.model, ','.join(args.outputs), ','.join(args.inputs)])
    # wandb.init(project="distribution-regression", name=name)
    wandb.init(project="blood-distribution", name=name)
    wandb.config.update(args)

    if args.id_file:
        id_file = args.id_file
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

    # dates inputs outputs
    train_data = FullSampleDataset(test=False, **data_config)
    test_data = FullSampleDataset(test=True, **data_config)
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
    
    model_params = {'n_outputs': len(args.outputs), 'n_inputs': len(args.inputs)}
    model = model_dict[args.model](**model_params)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)
    model = train_nn(model, args.name, optimizer, scheduler, train_generator, test_generator)
    if args.save:
        torch.save(model, args.name + '.pt')
