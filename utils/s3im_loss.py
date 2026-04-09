"""
S3IM: Stochastic Structural SIMilarity loss.

From the ICCV2023 paper:
    "S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields"
    https://github.com/Madaoer/S3IM-Neural-Fields

The key idea: randomly reshuffle flat ray samples into virtual 2D patches,
then compute SSIM on those patches. This captures structural similarity
without requiring spatially coherent patch sampling from the image.
"""

from utils.ssim import SSIM
import torch


class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (int): height of virtual patch(default: 64)
        patch_width (int): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)

    def forward(self, src_vec, tar_vec):
        """
        Compute S3IM loss between predicted and target ray colors.

        Args:
            src_vec: (N, 3) predicted RGB values for N rays
            tar_vec: (N, 3) ground truth RGB values for N rays
                     N must equal patch_height * patch_width.

        Returns:
            loss: scalar, 1 - SSIM of reshuffled virtual patches
        """
        assert tar_vec.shape[0] == self.patch_height * self.patch_width, (
            f"S3IM expects N={self.patch_height * self.patch_width} rays "
            f"(patch_height={self.patch_height} * patch_width={self.patch_width}), "
            f"but got N={tar_vec.shape[0]}."
        )
        device = tar_vec.device
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec), device=device)
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec), device=device)
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        return 1 - self.ssim_loss(src_patch, tar_patch)
