from typing import Dict, Union

import torch
from mmengine.model.utils import detect_anomalous_params
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS


def ddp_train_step_with_unit_module(self, data: Union[dict, tuple, list],
                                    optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    with optim_wrapper.optim_context(self):
        data, unit_losses = self.module.data_preprocessor(data, training=True)
        losses = self._run_forward(data, mode='loss')
    losses.update(unit_losses)
    parsed_loss, log_vars = self.module.parse_losses(losses)
    optim_wrapper.update_params(parsed_loss)
    if self.detect_anomalous_params:
        detect_anomalous_params(parsed_loss, model=self)
    return log_vars


# # switch MMDistributedDataParallel train_step and register it
MMDistributedDataParallel.train_step = ddp_train_step_with_unit_module
MODEL_WRAPPERS.register_module(module=MMDistributedDataParallel, force=True)

# # 创建自定义分布式数据并行类
# class CustomMMDDP(MMDistributedDataParallel):
#     def __init__(self,
#                 module,
#                 device_ids=None,
#                 output_device=None,
#                 dim=0,
#                 broadcast_buffers=True,
#                 process_group=None,
#                 bucket_cap_mb=25,
#                 find_unused_parameters=True,  # 关键修改点
#                 check_reduction=False,
#                 gradient_as_bucket_view=False,
#                 static_graph=False):
#         super().__init__(
#             module=module,
#             device_ids=device_ids,
#             output_device=output_device,
#             dim=dim,
#             broadcast_buffers=broadcast_buffers,
#             process_group=process_group,
#             bucket_cap_mb=bucket_cap_mb,
#             find_unused_parameters=find_unused_parameters,  # 强制启用参数检测
#             check_reduction=check_reduction,
#             gradient_as_bucket_view=gradient_as_bucket_view,
#             static_graph=static_graph
#         )
#
# # 覆盖原始方法并注册新类
# CustomMMDDP.train_step = ddp_train_step_with_unit_module
# MODEL_WRAPPERS.register_module(module=CustomMMDDP, name='CustomMMDDP', force=True)