import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS
from torch.nn.init import constant_
from mmdet.utils import ConfigType
from mmcv.cnn import build_activation_layer, build_norm_layer
from timm.models.layers import DropPath
import math

from .DCNv4_op.DCNv4.functions import DCNv4Function
from .DCNv4_op.DCNv4.modules import DCNv4


# -------------------------------------------------------------------------
# 分组可变形卷积
# -------------------------------------------------------------------------
class GroupDCNv4Layer(nn.Module):
	def __init__(
			self,
			channels=64,
			kernel_size=3,
			dw_kernel_size=None,
			stride=1,
			pad=1,
			dilation=1,
			groups=4,
			offset_scale=1.0,
			norm_cfg=dict(type='GN', num_groups=1, requires_grad=True),
			act_cfg=dict(type='GELU'),
			remove_center=False
	):
		super().__init__()
		if channels % groups != 0:
			raise ValueError(
				f"channels must be divisible by group, but got {channels} and {groups}"
			)
		_d_per_group = channels // groups
		dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
		# you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
		if not (1 << (_d_per_group - 1)).bit_length() == _d_per_group:
			warnings.warn(
				"You'd better set channels in DCNv4 to make the dimension of each attention head a power of 2 "
				"which is more efficient in our CUDA implementation."
			)

		self.channels = channels
		self.kernel_size = kernel_size
		self.dw_kernel_size = dw_kernel_size
		self.stride = stride
		self.dilation = dilation
		self.pad = pad
		self.group = groups
		self.group_channels = channels // groups
		self.offset_scale = offset_scale
		self.remove_center = int(remove_center)

		self.dw_conv = nn.Sequential(
			nn.Conv2d(
				channels,
				channels,
				kernel_size=dw_kernel_size,
				stride=1,
				padding=(dw_kernel_size - 1) // 2,
				groups=channels,
			),
			build_norm_layer(
				norm_cfg,
				channels,
			)[1],
			build_activation_layer(act_cfg),
		)
		self.offset_mask = nn.Linear(channels, int(
			math.ceil((groups * (kernel_size * kernel_size - self.remove_center) * 3) / 8) * 8))
		self._reset_parameters()

	def _reset_parameters(self):
		constant_(self.offset_mask.weight.data, 0.0)
		constant_(self.offset_mask.bias.data, 0.0)

	def forward(self, input):
		"""
		:param input: (N, C, H, W)
		:return output: (N, C, H, W)
		"""
		x = to_channels_last(input)
		N, H, W, _ = x.shape
		dtype = x.dtype

		x1 = input
		x1 = self.dw_conv(x1)
		x1 = to_channels_last(x1)

		# Calculate offset and mask in one go
		offset_mask = self.offset_mask(x1).reshape(N, H, W, -1)

		x = DCNv4Function.apply(
			x.contiguous(),
			offset_mask,
			self.kernel_size,
			self.kernel_size,
			self.stride,
			self.stride,
			self.pad,
			self.pad,
			self.dilation,
			self.dilation,
			self.group,
			self.group_channels,
			self.offset_scale,
			256,
			self.remove_center
		)

		x = to_channels_first(x)
		return x


def to_channels_last(tensor):
	return tensor.permute(0, 2, 3, 1).contiguous()


def to_channels_first(tensor):
	return tensor.permute(0, 3, 1, 2).contiguous()
#
#
# # ------------------------------------------------------------------------------
# # Feed Forward 模块
# # ------------------------------------------------------------------------------
class FeadForward(nn.Module):
	def __init__(
			self,
			in_channel: int = 64,
			mlp_ratio: float = 1.0,
			drop_rate: float = 0.1,
			act_cfg: ConfigType = dict(type="GELU"),
	):
		super(FeadForward, self).__init__()
		self.in_channel = in_channel
		self.mlp_ratio = mlp_ratio
		self.hidden_fetures = int(in_channel * mlp_ratio)

		self.input_project = nn.Conv2d(in_channel, self.hidden_fetures, kernel_size=1, bias=True)
		self.dwconv = nn.Conv2d(self.hidden_fetures, self.hidden_fetures, kernel_size=3, padding=1,
		                        groups=self.hidden_fetures, bias=True)

		self.output_project = nn.Conv2d(self.hidden_fetures, self.in_channel, kernel_size=1, bias=True)  # 1x1 conv

		self.act = build_activation_layer(act_cfg)

		self.dropout = nn.Dropout(drop_rate)

	def forward(self, x):
		"""

		:param input: [bs, C, H, W]
		:return: [bs, C, H, W]
		"""

		# feed forward
		x = self.input_project(x)
		x = self.act(x)
		x = self.dropout(x)
		x = self.dwconv(x)
		x = self.output_project(x)
		return x
#
#
# # ------------------------------------------------------------------------------
# # 参考Transformer模块的设计，使用DCN-v3替代多头注意力封装一个ModulatedDeformableConvModule
# # ------------------------------------------------------------------------------
@MODELS.register_module()
class GroupDCNv4(nn.Module):
	def __init__(
			self,
			channels: int = 128,
			groups: int = 4,
			kernel_size: int = 3,
			dilation: int = 1,
			mlp_ratio: float = 1.0,
			drop_rate: float = 0.1,
			norm_cfg: dict = dict(type='GN', num_groups=1, requires_grad=True),
			act_cfg: dict = dict(type="GELU"),
			drop_path: float = 0.,
			layer_scale=None,
	) -> None:
		super().__init__()
		# Normalization:
		_, self.norm1 = build_norm_layer(norm_cfg, channels)
		_, self.norm2 = build_norm_layer(norm_cfg, channels)

		# DCN-v4
		self.dcn_v4 = GroupDCNv4Layer(
			channels=channels,
			kernel_size=kernel_size,
			dilation=dilation,
			groups=groups,
			offset_scale=1.0,
			norm_cfg=norm_cfg,
			act_cfg=act_cfg,
		)

		from custom_modules import deformable_LKA_Attention
		self.dlka = deformable_LKA_Attention(d_model=channels) #jia
		# self.seam=SEAM(c1=channels)
		self.feed_forward = FeadForward(channels, mlp_ratio=mlp_ratio, drop_rate=drop_rate, act_cfg=act_cfg)

		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
		self.layer_scale = False
		if layer_scale is not None and type(layer_scale) in [int, float]:
			self.layer_scale = True
			self.gamma1 = nn.Parameter(
				layer_scale * torch.ones(1, channels, 1, 1), requires_grad=True
			)
			self.gamma2 = nn.Parameter(
				layer_scale * torch.ones(1, channels, 1, 1), requires_grad=True
			)

	def forward(self, x):
		if not self.layer_scale:
			shortcut = x
			x = self.norm1(x)
			x = self.dcn_v4(x)
			x=self.dlka(x)
			x = shortcut + self.drop_path(x)
			x = x + self.drop_path(self.feed_forward(self.norm2(x)))
			return x
		shortcut = x
		x = self.norm1(x)
		x = self.dcn_v4(x)
		x=self.dlka(x)
		x = shortcut + self.drop_path(self.gamma1 * x)
		x = x + self.drop_path(self.gamma2 * self.feed_forward(self.norm2(x)))
		return x




class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
    


class SEAM(nn.Module):
    def __init__(self, c1, n=1, reduction=16):
        super(SEAM, self).__init__()
        c2 = c1
        self.DCovN = nn.Sequential(
            # nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, groups=c1),
            # nn.GELU(),
            # nn.BatchNorm2d(c2),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1, groups=c2),
                    nn.GELU(),
                    nn.BatchNorm2d(c2)
                )),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.GELU(),
                nn.BatchNorm2d(c2)
            ) for i in range(n)]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()
        # self.initialize_layer(self.avg_pool)
        self.initialize_layer(self.fc)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.DCovN(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_layer(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)


def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2),
                nn.SiLU(),
                nn.BatchNorm2d(c2)
            )),
            nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
            nn.SiLU(),
            nn.BatchNorm2d(c2)
        ) for i in range(depth)]
    )
    return dcovn