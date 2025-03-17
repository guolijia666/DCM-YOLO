import torch
import torch.nn as nn
from mmyolo.registry import MODELS

@MODELS.register_module()
class SPD(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self,dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # print(f"Input shape before SPD: {x.shape}")
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

#SPD模块
# class SPD(nn.Module):  # SPD 层
#     """
#     这个模块实现了空间到深度的操作，它重新排列空间数据块到深度维度，
#     通过块大小增加通道数并减少空间维度。在卷积神经网络中常用此方法保持
#     下采样图像的高分辨率信息。
#     """
#     def __init__(self, block_size=2):
#         super(SPD, self).__init__()
#         self.block_size = block_size  # 块大小
#
#     def forward(self, x):
#         print(f"Input shape before SPD: {x.shape}")
#         N, C, H, W = x.size()  # 输入张量的维度
#         block_size = self.block_size  # 块大小
#
#         # 确保高度和宽度可以被 block_size 整除
#         assert H % block_size == 0 and W % block_size == 0, \
#             f"空间维度必须能被 block_size 整除。得到的 H: {H}, W: {W}"
#
#         # 将空间块重新排列到深度
#         x_reshaped = x.view(N, C, H // block_size, block_size, W // block_size, block_size)
#         x_permuted = x_reshaped.permute(0, 3, 5, 1, 2, 4).contiguous()
#         out = x_permuted.view(N, C * block_size ** 2, H // block_size, W // block_size)
#         return out
#
