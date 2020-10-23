import torch


def train(model, name, optimizer, scheduler, train_generator, test_generator, start_on_gpu=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    losses = []
    losses_ts = []
    lr = []
    for _ in range(1000):
        aux = []
        for x, y in train_generator:
            if not start_on_gpu:
                x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} train loss {lr}": loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(np.mean(aux))
        scheduler.step()
        aux = []
        for x, y in test_generator:
            if not start_on_gpu:
                x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} test loss {lr}": loss})
        losses_ts.append(np.mean(aux))
    return model