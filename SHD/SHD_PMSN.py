import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import h5py
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from lib import set_seed, setup_logging


SHD_INPUT_UNITS = 700
SHD_OUTPUT_CLASSES = 20
SHD_MAX_TIME = 1.4


def parse_args():
    parser = argparse.ArgumentParser(description='SHD PMSN Training')

    parser.add_argument('--lr', default=1e-2, type=float, help='Learning rate')
    parser.add_argument('--neuron_lr', '--neuron-lr', default=1e-3, type=float, help='PMSN neuron learning rate')
    parser.add_argument('--weight_decay', '--weight-decay', default=0, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--step_lr', '--step-lr', action='store_true', help='Use StepLR instead of cosine annealing')

    parser.add_argument('--batch_size', '--batch-size', default=40, type=int, help='Batch size')

    parser.add_argument('--d_input', '--input-dim', default=140, type=int, help='Input dimension')
    parser.add_argument('--d_model', '--hidden-dim', default=352, type=int, help='Model dimension')
    parser.add_argument('--d_state', '--d-state', default=4, type=int, help='PMSN state dimension')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout')
    parser.add_argument('--T', default=250, type=int, help='Number of time steps')

    parser.add_argument('--data_path', default='/path/', type=str, help='SHD dataset root path')
    parser.add_argument('--ckpt_path', '--ckpt-path', default='exp/shd', type=str, help='Checkpoint path')
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=1111, type=int)
    parser.add_argument('--log_interval', default=300, type=int)
    return parser.parse_args()


def count_para(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def build_model(args, device):
    from PMSN import SHDNet_ours

    return SHDNet_ours(
        bn=False,
        d_input=args.d_input,
        d_model=args.d_model,
        d_out=SHD_OUTPUT_CLASSES,
        d_state=args.d_state,
        dropout=args.dropout,
        lr=args.neuron_lr,
        T=args.T,
    ).to(device)


def setup_optimizer(model, lr, weight_decay, epochs, step_lr=False):
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

    if step_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, group in enumerate(optimizer.param_groups):
        group_hps = {k: group.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(group['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


def train(epoch, train_loader, optimizer, model, criterion, args):
    loss_record = []
    predict_tot = []
    label_tot = []
    model.train()
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device).long()
        optimizer.zero_grad()

        outputs = model(inputs.permute(0, 2, 1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        predict = torch.argmax(outputs, axis=1)
        loss_record.append(loss.detach().cpu())
        predict_tot.append(predict)
        label_tot.append(targets)

        if (batch_idx + 1) % args.log_interval == 0:
            print(
                '\nEpoch [%d/%d], Step [%d/%d], Loss: %.5f, Time elasped:%.2f'
                % (
                    epoch,
                    args.epochs + args.start_epoch,
                    batch_idx + 1,
                    len(train_loader) // args.batch_size,
                    loss_record[-1] / args.batch_size,
                ),
                time.time() - start_time,
            )

    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return train_acc, train_loss


def evaluate(data_loader, model, criterion, args):
    model.eval()
    with torch.no_grad():
        predict_tot = []
        label_tot = []
        loss_record = []

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device).long()
            outputs = model(inputs.permute(0, 2, 1))
            loss = criterion(outputs, targets)
            predict = torch.argmax(outputs, axis=1)

            loss_record.append(loss)
            predict_tot.append(predict)
            label_tot.append(targets)

        predict_tot = torch.cat(predict_tot)
        label_tot = torch.cat(label_tot)
        eval_acc = torch.mean((predict_tot == label_tot).float())
        eval_loss = torch.tensor(loss_record).sum() / len(label_tot)
        return eval_acc, eval_loss


class SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, device, shuffle=True):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.device = device
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.int64)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.counter >= self.number_of_batches:
            raise StopIteration

        batch_index = self.sample_index[
            self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)
        ]
        coo = [[] for _ in range(3)]
        for batch_pos, sample_idx in enumerate(batch_index):
            times = np.digitize(self.firing_times[sample_idx], self.time_bins)
            units = self.units_fired[sample_idx]
            batch = [batch_pos for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        indices = torch.LongTensor(coo).to(self.device)
        values = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)
        inputs = torch.sparse.FloatTensor(
            indices,
            values,
            torch.Size([len(batch_index), self.nb_steps, self.nb_units]),
        ).to_dense()
        targets = torch.tensor(self.labels_[batch_index], device=self.device)
        self.counter += 1
        return inputs.to(device=self.device), targets.to(device=self.device)


def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, device, shuffle=True):
    """Generate sparse SHD batches from an HDF5 spike dataset."""
    labels_ = np.array(y, dtype=np.int64)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))
    firing_times = X['times']
    units_fired = X['units']
    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for _ in range(3)]
        for batch_pos, sample_idx in enumerate(batch_index):
            times = np.digitize(firing_times[sample_idx], time_bins)
            units = units_fired[sample_idx]
            batch = [batch_pos for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        indices = torch.LongTensor(coo).to(device)
        values = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        inputs = torch.sparse.FloatTensor(indices, values, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        targets = torch.tensor(labels_[batch_index], device=device)

        yield inputs.to(device=device), targets.to(device=device)
        counter += 1


def get_data(data_path):
    train_file = h5py.File(os.path.join(data_path, 'shd_train.h5'), 'r')
    test_file = h5py.File(os.path.join(data_path, 'shd_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    return (x_train, y_train), (x_test, y_test)


def make_save_path(args):
    return os.path.join(
        args.ckpt_path,
        'lr' + str(args.lr)
        + '_nlr' + str(args.neuron_lr)
        + '_wd' + str(args.weight_decay)
        + '_drop_' + str(args.dropout)
        + '_d' + str(args.d_model)
        + '_s' + str(args.d_state)
        + '_seed' + str(args.seed),
    )


def main(args, model, save_path):
    args.start_epoch = 0
    best_acc = 0
    best_epoch = 0
    train_trace = {'acc': [], 'loss': [], 'acc_spk': [], 'loss_spk': []}
    val_trace = {'acc': [], 'loss': []}

    (x_train, y_train), (x_test, y_test) = get_data(args.data_path)
    train_loader = SpikeIterator(
        x_train, y_train, args.batch_size, args.T, SHD_INPUT_UNITS, max_time=SHD_MAX_TIME, device=args.device, shuffle=True
    )
    val_loader = SpikeIterator(
        x_test, y_test, args.batch_size, args.T, SHD_INPUT_UNITS, max_time=SHD_MAX_TIME, device=args.device, shuffle=False
    )

    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, step_lr=args.step_lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    if args.epochs == 0:
        logging.info('Skip training because epochs is 0.')
        return

    for epoch in tqdm(range(args.start_epoch, args.start_epoch + args.epochs)):
        train_loader.reset()
        train_acc, train_loss = train(epoch, train_loader, optimizer, model, criterion, args)

        train_loader.reset()
        train_acc_spk, train_loss_spk = evaluate(train_loader, model, criterion, args)

        scheduler.step()

        val_loader.reset()
        val_acc, val_loss = evaluate(val_loader, model, criterion, args)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': model.state_dict(),
                'traces': {'train': train_trace, 'val': val_trace},
                'config': args,
            }
            torch.save(state, os.path.join(save_path, 'checkpoint.pth'))

        train_trace['acc'].append(train_acc)
        train_trace['loss'].append(train_loss)
        train_trace['acc_spk'].append(train_acc_spk)
        train_trace['loss_spk'].append(train_loss_spk)
        val_trace['acc'].append(val_acc)
        val_trace['loss'].append(val_loss)

        logging.info(
            'Epoch %d: train acc %.5f, train acc with spike %.5f, test acc %.5f, best acc %.5f '
            % (epoch, train_acc, train_acc_spk, val_acc, best_acc)
        )

    train_loader.reset()
    train_acc, _ = train(epoch, train_loader, optimizer, model, criterion, args)
    logging.info(
        'Finish training: the final training accuracy is {} and the best validation accuray is {} in epoch {}. \n The relate checkpoint path: {}'.format(
            train_acc, best_acc, best_epoch, save_path
        )
    )


if __name__ == '__main__':
    args = parse_args()
    args.dataset = 'SHD'
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = str(device)

    model = build_model(args, device)
    set_seed(args.seed)

    save_path = make_save_path(args)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('saving to:' + str(save_path))
    logging.info('args:' + str(args))
    logging.info(str(model))
    logging.info(f"Parameter number: {count_para(model)}")

    main(args, model, save_path)
