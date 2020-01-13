import time
import torch
import torch.nn.functional as F


def print_info(info, log_fp=None):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, ACC: {:.4f}, '
               'Train Loss: {:.4f}, Test Loss:{:.4f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['acc'], info['train_loss'], info['test_loss'])
    print(message)
    if log_fp:
        with open(log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)


def run(model, train_loader, test_loader, target, num_nodes, epochs, optimizer,
        scheduler, device):

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, train_loader, target, optimizer, device)
        t_duration = time.time() - t
        scheduler.step()
        acc, test_loss = test(model, test_loader, num_nodes, target, device)
        eval_info = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'acc': acc,
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration
        }

        print_info(eval_info)


def train(model, train_loader, target, optimizer, device):
    model.train()

    total_loss = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x = data.x.to(device)
        loss = F.nll_loss(model(x), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, num_nodes, target, device):
    model.eval()
    correct = 0
    total_loss = 0
    n_graphs = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            out = model(data.x.to(device))
            total_loss += F.nll_loss(out, target).item()
            pred = out.max(1)[1]
            correct += pred.eq(target).sum().item()
            n_graphs += data.num_graphs
    return correct / (n_graphs * num_nodes), total_loss / len(test_loader)
