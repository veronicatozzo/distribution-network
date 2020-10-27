import torch


def train(model, name, optimizer, scheduler, train_generator, test_generator, start_on_gpu=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    step = 0
    best_loss = None
    for epoch in range(1000):
        aux = []
        for x, y in train_generator:
            if not start_on_gpu:
                x, y = x.to(device), y.to(device)
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
            if not start_on_gpu:
                x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            aux.append(loss.item())
            wandb.log({f"{name} test loss per step": loss}, step=step)
        test_loss = np.mean(aux)
        wandb.log({f"{name} test loss per epoch": test_loss}, step=epoch)
        if not best_loss or (test_loss > best_loss):
            wandb.run.summary["best_loss"] = test_loss
            best_loss = test_loss
    return model