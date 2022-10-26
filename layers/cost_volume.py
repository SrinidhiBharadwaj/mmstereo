# Copyright 2021 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

def cost_volume_left(left, right, num_disparities: int):
    batch_size, channels, height, width = left.shape

    outputs = []
    for i in range(num_disparities):
        cost = left[:, :, :, i:] * right[:, :, :, :width-i]
        zeros = torch.zeros((batch_size, channels, height, i), dtype=left.dtype, device=left.device)
        out = torch.cat([zeros, cost], dim=-1)
        outputs.append(out[:, :, None])

    return torch.cat(outputs, dim=2)


def cost_volume_right(left, right, num_disparities: int):
    batch_size, channels, height, width = left.shape

    outputs = []
    for i in range(num_disparities):
        zeros = torch.zeros((batch_size, channels, height, i), dtype=left.dtype, device=left.device)
        cost = left[:, :, :, i:] * right[:, :, :, :width-i]
        out = torch.cat([cost, zeros], dim=-1)
        outputs.append(out[:, :, None])

    return torch.cat(outputs, dim=2)


class CostVolume(nn.Module):
    """Compute cost volume using cross correlation of left and right feature maps"""

    def __init__(self, num_disparities, is_right=False):
        super().__init__()
        self.num_disparities = num_disparities
        self.is_right = is_right

    def forward(self, left, right):
        original_dtype = left.dtype
        left = left.to(torch.float32)
        right = right.to(torch.float32)
        if self.is_right:
            out = cost_volume_right(left, right, self.num_disparities)
        else:
            out = cost_volume_left(left, right, self.num_disparities)
        return torch.clamp(out, -1e3, 1e3).to(original_dtype)

