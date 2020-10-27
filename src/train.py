import wandb
import numpy as np
import torch
import torch.nn as nn


def train(model, name, optimizer, scheduler, train_generator, test_generator):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # by default, reduction = mean when multiple outputs
    criterion = nn.MSELoss()
    step = 0
    best_loss = None
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    for epoch in range(1000):
        aux = []
        for x, y in train_generator:
            x, y = x.type(dtype).to(device), y.type(dtype).to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} train loss per step": loss}, step=step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        wandb.log({f"{name} train loss per epoch": np.mean(aux)}, step=epoch)
        scheduler.step()

        aux = []
        for x, y in test_generator:
            x, y = x.type(dtype).to(device), y.type(dtype).to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} test loss per step": loss}, step=step)
        test_loss = np.mean(aux)
        wandb.log({f"{name} test loss per epoch": test_loss}, step=epoch)
        if not best_loss or (test_loss > best_loss):
            wandb.run.summary["best_loss"] = test_loss
            best_loss = test_loss
    return model
