from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS
from mmyolo.models import YOLOv5PAFPN
from typing import List, Union

import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig

# from custom_modules.DySample import DySample
# from custom_modules.SPD import SPD
from mmyolo.models import CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
# from custom_modules.YOLOV5PAFPN import YOLOV5PAFPN
from custom_modules import DySample


@MODELS.register_module()
class YOLOV8PAFPN(YOLOv5PAFPN):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 3.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_upsample_layer(self, idx, *args, **kwargs) -> nn.Module:
        """Build upsample layer."""
        in_channels = self.in_channels[idx]
        return DySample(in_channels=int(in_channels*0.5), scale=2)
        

    

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()
    

        

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """


        if idx == 2:
            return CSPLayerWithTwoConv(
                make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                               self.widen_factor),
                make_divisible(self.out_channels[idx - 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            from custom_modules import SEAM
            c2f_layer = CSPLayerWithTwoConv(
                make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                               self.widen_factor),
                make_divisible(self.out_channels[idx - 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

            seam_layer=SEAM(
                c1=make_divisible(self.out_channels[idx - 1], self.widen_factor)
            )

            return nn.Sequential(
                c2f_layer,
                seam_layer
            )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """

        from custom_modules import SEAM
        c2f_layer = CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        seam_layer=SEAM(
            c1=make_divisible(self.out_channels[idx + 1], self.widen_factor)
        )

        return nn.Sequential(
            c2f_layer,
            seam_layer
        )

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    