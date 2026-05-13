import math

import torch
import torch.nn as nn
from einops import rearrange


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, x):
        if not self.training:
            return x

        if not self.transposed:
            x = rearrange(x, 'b ... d -> b d ...')

        mask_shape = x.shape[:2] + (1,) * (x.ndim - 2) if self.tie else x.shape
        mask = torch.rand(*mask_shape, device=x.device) < 1.0 - self.p
        x = x * mask * (1.0 / (1 - self.p))

        if not self.transposed:
            x = rearrange(x, 'b d ... -> b ... d')
        return x


class PMSN_kernel(nn.Module):
    """Generate the diagonal SSM convolution kernel used by CIFAR32 PMSN neurons."""

    def __init__(self, d_model, N=64, dt_min=1e-3, dt_max=1e-1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H).uniform_(0, 1) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        self.register("log_dt", log_dt, lr)
        diag_indices = torch.arange(N)
        sub_diag_indices = diag_indices[:-1] + 1
        super_diag_indices = diag_indices[1:] - 1

        S = torch.zeros(N, N)
        S[diag_indices, diag_indices] = -0.5
        S[diag_indices[:-1], sub_diag_indices] = 5.0 * (torch.arange(N - 1) + 1)
        S[diag_indices[1:], super_diag_indices] = -5.0 * (torch.arange(N - 1) + 1)

        S_diag = torch.diagonal(S)
        A_real = (torch.mean(S_diag) * torch.ones_like(S_diag)).unsqueeze(0).repeat(H, 1)

        A_imag, V = torch.linalg.eigh(S * -1j)
        A_imag = A_imag.unsqueeze(0).repeat(H, 1)

        log_A_real = torch.log(-A_real)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr * 10)

        B = torch.ones(H, N)
        C = torch.zeros(H, N)
        C[:, -1] = 1
        Vinv = V.conj().T
        CV = torch.einsum('hm,mn->hn', C + 0j, V)
        VinvB = torch.einsum('mn,hn->hm', Vinv, B + 0j)

        self.register("VinvB_real", VinvB.real, lr * 10)
        self.register("VinvB_imag", VinvB.imag, lr * 10)
        self.register("CV_real", CV.real, lr * 10)
        self.register("CV_imag", CV.imag, lr * 10)

    def forward(self, L, u=None):
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        B = self.VinvB_real + 1j * self.VinvB_imag
        C = self.CV_real + self.CV_imag * 1j

        dt = torch.exp(self.log_dt)
        A_bar = torch.exp(A * dt.unsqueeze(-1))
        B_bar = (A_bar - 1) * B / A

        logK = (A * dt.unsqueeze(-1)).unsqueeze(-1) * torch.arange(L, device=A.device)
        K = torch.exp(logK)
        KB = torch.einsum('hnl,hn->hnl', K, B_bar)
        CKB = torch.einsum('hn,hnl->hl', C, KB).real
        return CKB

    def register(self, name, tensor, lr=None, weight_decay=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
            return

        self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0 if weight_decay is None else weight_decay}
        if lr is not None:
            optim["lr"] = lr
        setattr(getattr(self, name), "_optim", optim)


class PMSN_neuron(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=4,
        dropout=0.0,
        transposed=True,
        T=32,
        reset=True,
        cumsum=False,
        coeff=0.2,
        **kernel_args,
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.T = T
        self.reset = reset
        self.cumsum = cumsum
        self.coeff = coeff

        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = PMSN_kernel(self.h, N=self.n, **kernel_args)
        self.thresh = torch.tensor([1.0])

    def forward(self, u, **kwargs):
        input_ndim = u.ndim
        if input_ndim != 3:
            u = u.unsqueeze(-1)

        _, C, H = u.size()
        u = u.view(self.T, -1, C, H).permute(1, 2, 3, 0)

        k = self.kernel(L=self.T, u=u)
        k_f = torch.fft.rfft(k, n=2 * self.T)
        u_f = torch.fft.rfft(u, n=2 * self.T)
        uk_f = torch.einsum('bcht,ct->bcht', u_f, k_f)
        y = torch.fft.irfft(uk_f, n=2 * self.T)[..., :self.T]

        y = y + (u * self.D.unsqueeze(-1).unsqueeze(-1))
        y = self.IF_compensate(y * self.coeff)
        y = y.permute(3, 0, 1, 2).reshape(-1, C, H)

        if input_ndim != 3:
            y = y.squeeze(-1)
        return y

    def IF_compensate(self, x):
        if self.reset:
            return myfloor(x.relu(), self.thresh.to(x.device))
        if self.cumsum:
            return Triangle.apply(x.cumsum(-1), self.thresh.to(x.device))
        return Triangle.apply(x, self.thresh.to(x.device))


class StaticPMSN(nn.Module):
    def __init__(self, d_model, lr, d_state, T, reset, cumsum, coeff):
        super().__init__()
        self.neuron = PMSN_neuron(
            d_model=d_model,
            lr=lr,
            d_state=d_state,
            T=T,
            reset=reset,
            cumsum=cumsum,
            coeff=coeff,
        )

    def forward(self, x_seq):
        if x_seq.ndim == 4:
            TB, C, W, H = x_seq.size()
            output = self.neuron(x_seq.flatten(-2, -1))
            return output.view(TB, C, W, -1)
        if x_seq.ndim == 2:
            TB, _ = x_seq.size()
            output = self.neuron(x_seq.unsqueeze(-1))
            return output.view(TB, -1)
        raise NotImplementedError


class CIFAR32PMSNModel(nn.Module):
    def __init__(self, lr, d_state, args):
        super().__init__()
        self.lr = lr
        self.d_state = d_state
        self.reset = args.reset
        self.cumsum = args.cumsum
        self.coeff = args.coeff

        self.static_conv = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
        )
        self.conv1 = nn.Sequential(
            PMSN_neuron(128, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            PMSN_neuron(128, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            PMSN_neuron(128, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.AvgPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
        )
        self.conv2 = nn.Sequential(
            PMSN_neuron(128, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            PMSN_neuron(128, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            PMSN_neuron(128, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.AvgPool1d(2),
        )
        self.flat = nn.Flatten()
        self.dropout = DropoutNd(0.0)
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 128 * 2, bias=False),
            PMSN_neuron(128 * 2, lr=self.lr, d_state=self.d_state, reset=self.reset, cumsum=self.cumsum, coeff=self.coeff),
            nn.Linear(128 * 2, args.class_num, bias=False),
        )

    def forward(self, x):
        B, _, _, T = x.size()
        x = x.permute(3, 0, 1, 2).flatten(0, 1)
        x = self.static_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dropout(x.view(T, B, -1).permute(1, 2, 0)).permute(2, 0, 1).reshape(T * B, -1)
        x = self.fc(x)
        out_spikes_counter = x.view(T, B, -1)
        return out_spikes_counter.mean(dim=0)


class Triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh_t, gamma=1.0):
        out = input.gt(thresh_t).float()
        gamma_tensor = torch.tensor([gamma], device=input.device)
        ctx.save_for_backward(input, thresh_t, gamma_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, thresh_t, gamma_tensor = ctx.saved_tensors
        gamma = gamma_tensor[0].item()
        grad_input = grad_output.clone()
        grad = (1 / gamma) * (1 / gamma) * ((gamma - abs(input - thresh_t)).clamp(min=0))
        return grad_input * grad, None, None


class surrogate_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, gamma=1.0):
        cum_x = input.cumsum(dim=-1)
        cum_x_shift = cum_x.clone()
        cum_x_shift[..., 1:] = cum_x[..., :-1]
        cum_x_shift[..., 0] = 0
        spike_shift = (cum_x_shift / thresh).floor().clamp(min=0)

        if burst:
            out = ((cum_x - spike_shift * thresh) / thresh).floor().clamp(min=0)
        else:
            out = ((cum_x - spike_shift * thresh) / thresh).floor().clamp(min=0, max=1)

        gamma_tensor = torch.tensor([gamma], device=input.device)
        ctx.save_for_backward(thresh, cum_x - spike_shift * thresh, gamma_tensor)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        thresh, delta, gamma_tensor = ctx.saved_tensors
        gamma = gamma_tensor[0].item()
        grad_input = grad_output.clone()

        if burst:
            grad = (1 / gamma) * (1 / gamma) * ((gamma - abs(delta - thresh) % thresh).clamp(min=0))
        else:
            grad = (1 / gamma) * (1 / gamma) * ((gamma - abs(delta - thresh)).clamp(min=0))

        return grad_input * grad, None, None


burst = False
myfloor = surrogate_grad.apply
