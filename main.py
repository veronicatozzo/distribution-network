import os
import argparse
import datetime
import wandb
import torch
from torch.data import DataLoader

from deep_sets.models import SmallDeepSetMax, SmallDeepSetMean, SmallDeepSetSum
from set_transformer.models import SmallSetTransformer
from src.dataset import FullSampleDataset
from src.train import train


os["WANDB_API_KEY"] = "893130108141453e3e50e00010d3e3fced11c1e8"

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('inputs', metavar='N', type=str, nargs=',',
                    help='a list of strings denoting the distribution inputs')
parser.add_argument('outputs', metavar='N', type=str, nargs=',',
                    help='a list of strings denoting the prediction outputs')
parser.add_argument('model', metavar='N', type=str, nargs=',',
                    help='a list of strings denoting the prediction outputs')

parser.add_argument('--id_file', type=str, default='', help='filename of the ids to use')
parser.add_argument('--num_subsamples', type=int, help='number of samples to use in each distribution')
parser.add_argument('--permute_subsamples', dest='permute_subsamples', action='store_true')
parser.add_argument('--normalizer', type=str, help='name of the normalizer')

parser.add_argument('model', metavar='N', type=str, nargs=',', default='',
                    help='a list of strings denoting the prediction outputs')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, help='number of cpu workers in the data loader')

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

if __name__ == "__main__":
    if args.name:
        name = args.name
    else:
        name = '_'.join(args.model, args.outputs, args.inputs))
    wandb.init(project="distribution-regression", name=name)
    wandb.config.update(args)

    inputs = args.inputs.split(',')
    outputs = args.outputs.split(',')
    if args.id_file:
        id_file = args.id_file
    else:
        id_file = str("id_files/{:%B-%d-%Y}.txt".format(datetime.now()))

    data_config = {
        'inputs': inputs,
        'outputs': outputs,
        'model': args.model,
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
    
    model_params = {'n_outputs': len(outputs)}
    model = model_dict[args.model](**model_params)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)
    model = train(model, args.name, optimizer, scheduler, train_generator, test_generator)
    if args.save:
        torch.save(model, args.name + '.pt')