import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from autoaugment import ClassificationPresetTrain, RandomCutmix, RandomMixup
from PMSN import CIFAR32PMSNModel
from lib import count_parameters, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--neuron_lr', default=1e-3, type=float, help='PMSN neuron learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='Training epochs')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar100', 'cifar10'], type=str, help='Dataset')
    parser.add_argument('--data_path', default='/path/', type=str, help='Dataset root path')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')

    parser.add_argument('--d_state', default=4, type=int, help='PMSN state dimension')
    parser.add_argument('--T', default=4, type=int, help='Number of time steps')
    parser.add_argument('--reset', action='store_true', help='Enable self-defined reset mechanism')
    parser.add_argument('--cumsum', action='store_true', help='Enable cumsum')
    parser.add_argument('--coeff', default=0.2, type=float, help='PMSN input coefficient')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint')
    parser.add_argument('--name', default='', type=str, help='Name of experiment')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--seed', default=1234, type=int, help='Seed for initializing training')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_save_path(args):
    save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return save_path + '_' + args.name + '_' + str(args.seed)


def get_transforms(args):
    if args.dataset == 'cifar10':
        transform_train = ClassificationPresetTrain(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
            interpolation=InterpolationMode('bilinear'),
            auto_augment_policy='ta_wide',
            random_erase_prob=0.1,
        )
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'cifar100':
        transform_train = ClassificationPresetTrain(
            mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            interpolation=InterpolationMode('bilinear'),
            auto_augment_policy='ta_wide',
            random_erase_prob=0.1,
        )
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            ),
        ])
    else:
        raise NotImplementedError(args.dataset)

    return transform_train, transform_test


def get_data(args):
    transform_train, transform_test = get_transforms(args)

    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=args.data_path, train=True, transform=transform_train, download=True)
        test_set = torchvision.datasets.CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
    elif args.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root=args.data_path, train=True, transform=transform_train, download=True)
        test_set = torchvision.datasets.CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)
    else:
        raise NotImplementedError(args.dataset)

    mixupcutmix = torchvision.transforms.RandomChoice([
        RandomMixup(args.class_num, p=1.0, alpha=0.2),
        RandomCutmix(args.class_num, p=1.0, alpha=1.0),
    ])
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_model(args):
    return CIFAR32PMSNModel(lr=args.neuron_lr, d_state=args.d_state, args=args)


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
        optimizer.add_param_group({"params": params, **hp})

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, group in enumerate(optimizer.param_groups):
        group_hps = {k: group.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(group['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


def load_checkpoint(model, resume_path):
    if not resume_path:
        return 0

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(resume_path)
    msg = model.load_state_dict(checkpoint['model'])
    print('missing:' + str(set(msg.missing_keys)))
    best_acc = checkpoint['acc']
    print(f"best_acc: {best_acc}")
    return best_acc


def train(model, train_loader, optimizer, criterion, device, args):
    model.train()
    train_loss = 0
    train_loss2 = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(train_loader), mininterval=20)

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets, label_smoothing=0.1)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.argmax(1)).sum().item()

        if batch_idx % 50 == 0:
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Loss2: %.3f |Train Acc: %.3f%% (%d/%d)'
                % (
                    batch_idx,
                    len(train_loader),
                    train_loss / (batch_idx + 1),
                    train_loss2 / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                )
            )


def evaluate(epoch, dataloader, model, criterion, device, save_path, best_acc, checkpoint=False):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), mininterval=20)
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (batch_idx, len(dataloader), eval_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
                )

    acc = 100.0 * correct / total
    if checkpoint and acc > best_acc:
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_path, 'ckpt.pth'))
        best_acc = acc
    return acc, best_acc


def main():
    args = parse_args()
    args.class_num = 10 if args.dataset == 'cifar10' else 100

    save_path = make_save_path(args)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('saving to:' + str(save_path))
    logging.info('args:' + str(args))

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f'==> Preparing {args.dataset} data..')
    logging.info(
        'batch_size = %d, learning_rate = %.4f, d_state = %d'
        % (args.batch_size, args.lr, args.d_state)
    )
    train_loader, test_loader = get_data(args)

    logging.info('==> Building model..')
    model = build_model(args)
    logging.info(model)
    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    best_acc = load_checkpoint(model, args.resume)
    logging.info(f"para: {count_parameters(model)}")

    criterion = nn.functional.cross_entropy
    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
    )

    results = dict()
    pbar = tqdm(range(0, args.epochs))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description('Epoch: %d' % epoch)
        else:
            pbar.set_description('Epoch: %d | Val acc: %1.3f | Best acc: %1.3f' % (epoch, val_acc, best_acc))
            logging.info('Epoch: %d | Val acc: %1.3f | Best acc: %1.3f' % (epoch, val_acc, best_acc))

        if args.evaluate is False:
            train(model, train_loader, optimizer, criterion, device, args)

        val_acc, best_acc = evaluate(
            epoch, test_loader, model, criterion, device, save_path, best_acc, checkpoint=True
        )
        results[str(epoch)] = val_acc
        print(f"val_acc {val_acc}")
        torch.save(results, os.path.join(save_path, 'results'))

        if args.evaluate:
            exit()
        scheduler.step()


if __name__ == "__main__":
    main()
