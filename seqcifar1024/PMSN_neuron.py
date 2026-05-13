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
        """Input shape: (B, H, L)."""
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
    """Generate the diagonal SSM convolution kernel for PMSN."""

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

        self.register("VinvB_real", VinvB.real, lr * 10, weight_decay=lr * 10)
        self.register("VinvB_imag", VinvB.imag, lr * 10)
        self.register("CV_real", CV.real, lr * 10)
        self.register("CV_imag", CV.imag, lr * 10)

    def forward(self, L, u=None, mode='parallel'):
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        B = self.VinvB_real + 1j * self.VinvB_imag
        C = self.CV_real + self.CV_imag * 1j

        dt = torch.exp(self.log_dt)
        A_bar = torch.exp(A * dt.unsqueeze(-1))
        B_bar = (A_bar - 1) * B / A

        if mode == 'parallel':
            logK = (A * dt.unsqueeze(-1)).unsqueeze(-1) * torch.arange(L, device=A.device)
            K = torch.exp(logK)
            KB = torch.einsum('hnl,hn->hnl', K, B_bar)
            CKB = torch.einsum('hn,hnl->hl', C, KB).real
            return CKB
        if mode == 'serial':
            return C, A_bar, B_bar
        raise NotImplementedError(mode)

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
    def __init__(self, d_model, d_state=4, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.decay = 0.5
        self.T = 784
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = PMSN_kernel(self.h, N=self.n, **kernel_args)
        self.dropout = DropoutNd(dropout / 5) if dropout > 0.0 else nn.Identity()
        self.thresh = torch.tensor([1.0])

    def forward(self, u, s_res=None, mode='parallel', **kwargs):
        """Input and output shape: (B, H, L)."""
        _, _, L = u.size()
        if mode == 'parallel':
            k = self.kernel(L=L, u=u, mode=mode)
            k_f = torch.fft.rfft(k, n=2 * L)
            u_f = torch.fft.rfft(u, n=2 * L)
            y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]

            y = y + (u * self.D.unsqueeze(-1))
            y = self.IF_compensate(y)
            spike = y + s_res if s_res is not None else y
            spike = self.dropout(spike)
            return spike, spike

        if mode == 'serial':
            C, A_bar, B_bar = self.kernel(L=L, u=u, mode=mode)
            spikes = []
            vs = 0
            C = C.unsqueeze(0)
            A_bar = A_bar.unsqueeze(0)
            B_bar = B_bar.unsqueeze(0)
            D = self.D.unsqueeze(0)
            thresh = self.thresh.to(u.device)
            for t in range(L):
                u_t = u[..., t]
                Bu = B_bar * u_t.unsqueeze(-1)
                x = A_bar * x + Bu if t > 0 else Bu
                Cx = (C * x).sum(dim=-1).real
                y = Cx + u_t * D
                spike_t, vs = self.serial_reset(y, vs, thresh)
                spikes.append(spike_t)
            spike = torch.stack(spikes, dim=-1)
            if s_res is not None:
                spike = spike + s_res
            spike = self.dropout(spike)
            return spike, spike

        raise NotImplementedError(mode)

    def IF_compensate(self, x):
        return myfloor(x.relu(), self.thresh.to(x.device))

    def serial_reset(self, x, mem, thresh):
        mem = mem + x.relu()
        spike = Triangle.apply(mem, thresh)
        reset_mem = (torch.floor(mem / thresh) * thresh - mem).detach() + mem
        mem = mem - reset_mem
        return spike, mem


class Triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh_t, gamma=1.0):
        out = input.gt(thresh_t).float()
        ctx.gamma = gamma
        ctx.save_for_backward(input, thresh_t)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, thresh_t = ctx.saved_tensors
        gamma = ctx.gamma
        grad_input = grad_output.clone()
        grad = (1 / gamma) * (1 / gamma) * ((gamma - abs(input - thresh_t)).clamp(min=0))
        return grad_input * grad, None, None


burst = False


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


myfloor = surrogate_grad.apply
