import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDySample(nn.Module):
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        assert scale in [2, 4], "Scale must be 2 or 4"
        self.scale = scale

        self.offset_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, 2 * scale * scale, 1, bias=False),
        )
        self.offset_conv[-1].weight.data.zero_()

        self.refine = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1, groups=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        self.upsample_static = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.max_offset = 1.0
        self.static_proj = nn.Conv2d(c1, c2, 1, 1, 0) if c1 != c2 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        static_feat = self.static_proj(self.upsample_static(x))

        offset = self.offset_conv(x)
        offset = torch.tanh(offset) * self.max_offset
        offset = F.pixel_shuffle(offset, self.scale)

        grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, H * self.scale, device=x.device),
            torch.linspace(-1, 1, W * self.scale, device=x.device),
            indexing='ij',
        )
        base_grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        dynamic_grid = base_grid + offset.permute(0, 2, 3, 1) * 0.1
        dynamic_grid = dynamic_grid.type_as(x)

        dynamic_feat = F.grid_sample(x, dynamic_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        dynamic_feat = self.refine(dynamic_feat)

        if self.training:
            self._log_offset_mean = offset.mean().item()
            self._log_offset_std = offset.std().item()
            self._log_offset_max = offset.max().item()
            self._log_alpha = self.alpha.item()

        return self.alpha * dynamic_feat + (1 - self.alpha) * static_feat
