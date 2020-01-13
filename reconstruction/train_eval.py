import time
import os
import torch
import torch.nn.functional as F


def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer,
        device):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        x = data.x.to(device)
        out = model(x)
        loss = F.l1_loss(out, x, reduction='mean')
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.x.to(device)
            pred = model(x)
            total_loss += F.l1_loss(pred, x, reduction='mean')
    return total_loss / len(loader)


def eval_error(model, test_loader, device, meshdata, out_dir):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = 'Error: {:.3f}+{:.3f} | {:.3f}'.format(mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/euc_errors.txt'
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
