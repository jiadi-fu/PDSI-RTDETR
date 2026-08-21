import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        var_term = x_minus_mu.pow(2).sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y = x_minus_mu.pow(2) / (4 * var_term) + 0.5
        return x * self.sigmoid(y)


class ConvSimAMPP(nn.Module):
    def __init__(self, c1, c2, e_lambda=1e-4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        c = c2
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.simam = SimAM(e_lambda)
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_proj = self.proj(x)
        local_feat = self.act(self.bn(self.dw(x_proj)))
        sa_out = self.simam(local_feat)
        if self.training:
            self._log_gamma = self.gamma.item()
            self._log_sa_mean = sa_out.mean().item()
        return self.gamma * sa_out + (1 - self.gamma) * x_proj
