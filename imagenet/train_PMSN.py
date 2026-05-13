import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import math
from torch.cuda import amp
import torch.distributed.optim
import argparse
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate
import sew_resnet, utils

class IFNode5(nn.Module):
    def __init__(self, T: int, surrogate_function: surrogate.SurrogateFunctionBase):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.fc = nn.Linear(T, T)
        nn.init.constant_(self.fc.bias, -1)

    def forward(self, x_seq: torch.Tensor):
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq.flatten(1))
        spike = self.surrogate_function(h_seq)
        return spike.view(x_seq.shape)

class PMSN(nn.Module):
    def __init__(self, T:int, d_model:int, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = 4
        self.d_output = self.h
        self.decay = 0.5
        self.T = T
        self.coeff = 1.0

        self.D = nn.Parameter(torch.rand(self.h))

        self.kernel = PMSN_kernel(self.h, N=self.n, **kernel_args)

        self.thresh = torch.tensor([1])

    def forward(self, u, **kwargs):

        T,B,C,H,_=u.size()

        u=u.view(T,B,C,-1).permute(1,2,3,0)  # [B, C, H, T]
        

        k = self.kernel(L=self.T) # (C L)

        k_f = torch.fft.rfft(k, n=2 * self.T)  # (C L)
        u_f = torch.fft.rfft(u, n=2 * self.T)  # (B C H L)
        uk_f = torch.einsum('bcht,ct->bcht', u_f, k_f)
        y = torch.fft.irfft(uk_f, n=2 * self.T)[..., :self.T]  # (B C H L)
        

        mem = y + (u * self.D.unsqueeze(-1).unsqueeze(-1).to(u.dtype))
        y = self.IF_compensate(mem)
        y = y.permute(3, 0, 1, 2).reshape(T,B,C,H,-1)
        

        return y

    def IF_compensate(self, x):
        spike = surrogate_grad.apply(x.relu(), self.thresh.to(x.device))

        return spike
class PMSN_kernel(nn.Module):

    def __init__(self, d_model, N=64, dt_min=2e-1, dt_max=1e-0, lr=None, **kwargs):
        super().__init__()
        wd=args.weight_decay
        H = d_model
        log_dt = torch.rand(H).uniform_(0, 1) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)  # [H]

        self.register("log_dt", log_dt, lr,weight_decay=wd)
        diag_indices = torch.arange(N)
        sub_diag_indices = diag_indices[:-1] + 1
        super_diag_indices = diag_indices[1:] - 1


        S = torch.zeros(N, N)
        S[diag_indices, diag_indices] = -0.5
        S[diag_indices[:-1], sub_diag_indices] = 0.25 * ((torch.arange(N - 1) + 1))
        S[diag_indices[1:], super_diag_indices] = -0.25 * ((torch.arange(N - 1) + 1))  
        S_diag = torch.diagonal(S)
        A_real = (torch.mean(S_diag) * torch.ones_like(S_diag)).unsqueeze(0).repeat(H, 1)
        A_imag, V = torch.linalg.eigh(S * -1j)  # [N; N,N]
        
        A_imag = A_imag.unsqueeze(0).repeat(H, 1)

        log_A_real = torch.log(-A_real)
        self.register("log_A_real", log_A_real,lr,weight_decay=wd)
        self.register("A_imag", A_imag, lr*10, weight_decay=wd)  # [H,N]

        B = torch.ones(H, N)

        C = torch.zeros(H, N)
        C[:, -1] = 1
        Vinv = V.conj().T  # [N,N]
        CV = torch.einsum('hm,mn->hn', C + 0j, V)  # [H,N]
        VinvB = torch.einsum('mn,hn->hm', Vinv, B + 0j)  # [H,N]

        VinvB_real = VinvB.real
        VinvB_img = VinvB.imag
        self.register("VinvB_real", VinvB_real, lr,weight_decay=wd)
        self.register("VinvB_imag", VinvB_img, lr,weight_decay=wd)

        CV_real = CV.real
        CV_img = CV.imag
        self.register("CV_real", CV_real, lr,weight_decay=wd)
        self.register("CV_imag", CV_img, lr,weight_decay=wd)


    def forward(self, L, u=None, **kwargs):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        B = self.VinvB_real + 1j * self.VinvB_imag  # (H,N)
        C = self.CV_real + self.CV_imag * 1j


        dt = torch.exp(self.log_dt)  # (H,1)
        A_bar = torch.exp(A * dt.unsqueeze(-1))  # [H N]
        B_bar = (A_bar - 1) * B / A


        logK = (A * dt.unsqueeze(-1)).unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)   e-At
        K = torch.exp(logK)
        KB = torch.einsum('hnl,hn->hnl', K, B_bar)  # e-At*B  # (H N L)
        CKB = torch.einsum('hn, hnl -> hl', C, KB).real  # (H L)

        return CKB

    def register(self, name, tensor, lr=None, weight_decay=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            if weight_decay is not None:
                optim = {"weight_decay": weight_decay}
            else:
                optim = {"weight_decay": 0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class surrogate_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, gamma=1.):
        cum_x = input.cumsum(dim=-1)
        cum_x_shift = cum_x.clone()
        cum_x_shift[..., 1:] = cum_x[..., :-1]
        cum_x_shift[..., 0] = 0
        spike_shift = (cum_x_shift / thresh).floor().clamp(min=0)
        out = ((cum_x - spike_shift * thresh) / thresh).floor().clamp(min=0, max=1)
        L = torch.tensor([gamma])
        ctx.save_for_backward(thresh, cum_x - spike_shift * thresh, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (thresh, delta, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - abs(delta - thresh)).clamp(min=0))
        grad_output = grad_input * tmp
        return grad_output, None
class Triangle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thresh_t, gamma=1.):
        out = input.gt(thresh_t).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, thresh_t, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, thresh_t, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - abs(input - thresh_t)).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

import numpy as np
np.random.seed(_seed_)


def setup_optimizer(model, lr, weight_decay,momentum):

    all_parameters = list(model.parameters())

    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    if args.adamw:
      optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
      optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
                             f"Optimizer group {i}",
                             f"{len(g['params'])} tensors",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        if output.dim() == 3:
            with torch.no_grad():
                output = output.mean(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)
            if output.dim() == 3:
                with torch.no_grad():
                    output = output.mean(0)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, cache_dataset, distributed):
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):


    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None


    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'new2_pmsn_{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    if args.cos_lr_T == -1:
        args.cos_lr_T = args.epochs

    output_dir += f'_coslr{args.cos_lr_T}'

    if args.adamw:
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    output_dir += f'_{args.world_size}gpu'

    if args.load is not None:
        output_dir += '_load'

    if args.tet:
        output_dir += '_tet'


    if output_dir:
        utils.mkdir(output_dir)


    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")

    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](pretrained=False, cnf='ADD', spiking_neuron=PMSN,
                                                surrogate_function=surrogate.ATan(), T=args.T, lr=args.neuron_lr)



    else:
        raise NotImplementedError(args.model)



    if args.load is not None:
        model.load_state_dict(torch.load(args.load), strict=False)
        print('load', args.load)


    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.tet:
        criterion = functional.temporal_efficient_training_cross_entropy
    else:
        def ce_loss(y, target):
            return F.cross_entropy(y.mean(0), target)
        criterion = ce_loss


    optimizer = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    
    if args.steplr:
      lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    else:
      lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)



    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True



        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth'))
            save_flag = False

            if epoch % 64 == 0 or epoch == args.epochs - 1:
                save_flag = True

            elif args.cos_lr_T == 0:
                for item in args.lr_step_size:
                    if (epoch + 2) % item == 0:
                        save_flag = True
                        break

            if save_flag:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/datasets/imagenet', help='dataset')

    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=320, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--neuron_lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')


    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')


    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adamw', action='store_true',
                        help='Use AdamW. The default optimizer is SGD.')

    parser.add_argument('--cos_lr_T', default=-1, type=int,
                        help='T_max of CosineAnnealingLR.')

    parser.add_argument('--load', type=str, default=None, help='the pt file path for loading pre-trained ANN weights')
    
    parser.add_argument('--tet', action='store_true', help='use the tet loss')
    parser.add_argument('--steplr', action='store_true', help='use the tet loss')



    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

