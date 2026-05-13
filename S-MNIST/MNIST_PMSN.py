import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
import time
import torchvision
import torchvision.transforms as transforms
from lib import set_seed, setup_logging, AverageMeter, ProgressMeter, accuracy, save_checkpoint
import os
import argparse
import logging
from PMSN_neuron import PMSN_neuron
import numpy as np



parser = argparse.ArgumentParser(description='S-MNIST/PS-MNIST Training')
parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
parser.add_argument('--neuron_lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay')
parser.add_argument('--epochs', default=200, type=int, help='Training epochs')
parser.add_argument('--name', default='PMSN', type=str, help='Name of model')
parser.add_argument('--permute', action='store_true', default=False, help='use psmnist or smnist')
parser.add_argument('--data_path', default='/path/', type=str, help='Dataset root path')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--d_state', default=4, type=int, help='Hidden dimension')
parser.add_argument('--dropout', default=0.02, type=float, help='Dropout')
parser.add_argument('--mode', default='parallel', choices=['parallel', 'serial'], help='PMSN computation mode')
parser.add_argument('--prenorm', default=True, action='store_true', help='Prenorm')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--norm', default='BN', choices=['BN', 'LN', 'None'], help='norm type: BN or LN or None')

set_seed(1111)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Preparing mnist data..')
print('batch_size = %d, learning_rate = %.4f, d_model = %d, d_state = %d, n_layer= %d, norm = %s, dropout = %.2f, mode = %s' %
      ( args.batch_size, args.lr, args.d_model, args.d_state,  args.n_layers, args.norm, args.dropout, args.mode))

def count_para(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 784).t())
])
trainset = torchvision.datasets.MNIST(
    root=args.data_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(
    root=args.data_path, train=False, download=True, transform=transform)

d_input = 1
d_output = 10

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def make_norm(d_model):
    if args.norm == 'BN':
        return nn.BatchNorm1d(d_model)
    if args.norm == 'LN':
        return nn.LayerNorm(d_model)
    if args.norm == 'None':
        return nn.Identity()
    raise NotImplementedError(args.norm)


class PMSNModel(nn.Module):
    def __init__(
        self,
        d_input,
        d_state,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=True,
    ):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.PMSN_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for n in range(n_layers):
            self.PMSN_layers.append(
                PMSN_neuron(d_model, d_state=d_state, dropout=dropout, lr=args.neuron_lr)
            )
            self.norms.append(make_norm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))
            if n < n_layers - 1:
                self.linear_layers.append(nn.Conv1d(d_model, d_model, kernel_size=1))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        spike = 0
        for idx, (layer, norm, dropout) in enumerate(zip(self.PMSN_layers, self.norms, self.dropouts)):
            z = x
            if self.prenorm:
                if args.norm in ['BN', 'None']:
                    z = norm(z)
                elif args.norm == 'LN':
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, spike = layer(z, spike, mode=args.mode)

            if idx < len(self.PMSN_layers) - 1:
                z = self.linear_layers[idx](z)
            z = dropout(z)
            x = z

        x = x.transpose(-1, -2)

        output = x.mean(dim=1)

        output = self.decoder(output)  # (B, d_model) -> (B, d_output)

        return output

print('==> Building model..')
model = PMSNModel(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
    d_state=args.d_state
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):

    all_parameters = list(model.parameters())

    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)


permute = torch.Tensor(np.random.permutation(int(784)).astype(np.float64)).long()
def train(epoch):
    epoch_start = time.time()
    max_batch_peak_memory_gb = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        if args.permute:
            inputs = inputs[:, permute, :]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], targets.size(0))
        top5.update(acc5[0], targets.size(0))
        losses.update(loss.item(), targets.size(0))


        torch.cuda.synchronize()
        batch_peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        max_batch_peak_memory_gb = max(max_batch_peak_memory_gb, batch_peak_memory_gb)


        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 300 == 0:
            print('Batch peak CUDA memory %.6f GB' % batch_peak_memory_gb)
            progress.display(batch_idx + 1)
    epoch_time = time.time() - epoch_start
    print('Train epoch time %.3f seconds | Train max batch peak CUDA memory %.6f GB' % (epoch_time, max_batch_peak_memory_gb))
    return top1.avg, epoch_time, max_batch_peak_memory_gb


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    eval_start = time.time()
    max_batch_peak_memory_gb = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.permute:
                inputs = inputs[:, permute, :]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], targets.size(0))
            top5.update(acc5[0], targets.size(0))
            losses.update(loss.item(), targets.size(0))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                batch_peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
                max_batch_peak_memory_gb = max(max_batch_peak_memory_gb, batch_peak_memory_gb)
            else:
                batch_peak_memory_gb = 0.0

            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx + 1) % 300 == 0:
                print('Inference batch peak CUDA memory %.6f GB' % batch_peak_memory_gb)
                progress.display(batch_idx + 1)
    eval_time = time.time() - eval_start
    print('Inference time %.3f seconds | Inference max batch peak CUDA memory %.6f GB' % (eval_time, max_batch_peak_memory_gb))

    if checkpoint:
        acc = top1.avg
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(save_path, 'checkpoint.pth.tar'), save_path=save_path)

    return top1.avg, best_acc, eval_time, max_batch_peak_memory_gb




if __name__ == "__main__":

    save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = 'exp/mnist/' + args.name + '_' + args.mode + '_l' +  str(args.n_layers) + '_h' +  str(args.d_model) + '_s'  + str(args.d_state) + '_drop'  + str(args.dropout) + '_' + save_path
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('saving to:' + str(save_path))
    logging.info('args:' + str(args))

    logging.info(str(model))
    para = count_para(model)
    logging.info(f"Parameter number: {para}")
    for epoch in range(start_epoch, args.epochs):
        train_acc, train_time, train_peak_memory = train(epoch)
        test_acc, best_acc, infer_time, infer_peak_memory = eval(epoch, testloader, checkpoint=True)
        scheduler.step()
        out_string = 'Epoch: %d | Train acc: %1.3f | Test acc: %1.3f | Best acc: %1.3f | Train time: %.3fs | Train max batch peak CUDA memory: %.6f GB | Inference time: %.3fs | Inference max batch peak CUDA memory: %.6f GB \n' % (epoch, train_acc, test_acc, best_acc, train_time, train_peak_memory, infer_time, infer_peak_memory)
        logging.info(out_string)
