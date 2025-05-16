import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class RandomPatchImageProcessor(nn.Module):
    def __init__(self, init_image: Tensor, patch_size: Tuple[int, int] = (50, 50)):
        super().__init__()
        self.init_image = init_image.detach()
        self.patch_size = patch_size

        _, _, H, W = init_image.shape
        C = init_image.shape[1]
        self.learned_patch = nn.Parameter(torch.rand(1, C, *patch_size))
        self.init_patch = self.learned_patch.data.clone()

    def forward(self, images: Tensor) -> Tensor:
        B, C, H, W = images.shape
        ph, pw = self.patch_size
        device = images.device

        # случайные координаты для каждого изображения в батче
        top = torch.randint(0, H - ph + 1, (B,), device=device)
        left = torch.randint(0, W - pw + 1, (B,), device=device)

        patched_images = images.clone()
        for i in range(B):
            patched_images[i, :, top[i]:top[i]+ph, left[i]:left[i]+pw] = self.learned_patch[0]

        return patched_images
