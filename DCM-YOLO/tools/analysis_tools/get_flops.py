# import argparse
# import tempfile
# from functools import partial
# from pathlib import Path
#
# import torch
# from mmengine.config import Config, DictAction
# from mmengine.logging import MMLogger
# from mmengine.model import revert_sync_batchnorm
# from mmengine.registry import init_default_scope
# from mmengine.runner import Runner
#
# from mmdet.registry import MODELS
#
# try:
#     from mmengine.analysis import get_model_complexity_info
#     from mmengine.analysis.print_helper import _format_size
# except ImportError:
#     raise ImportError('Please upgrade mmengine >= 0.6.0')
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Get a detector flops')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument(
#         '--shape',
#         type=int,
#         nargs='+',
#         default=[640, 640],
#         help='input image size')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     args = parser.parse_args()
#     return args
#
#
# def inference(args, logger):
#     if str(torch.__version__) < '1.12':
#         logger.warning(
#             'Some config files, such as configs/yolact and configs/detectors,'
#             'may have compatibility issues with torch.jit when torch<1.12. '
#             'If you want to calculate flops for these models, '
#             'please make sure your pytorch version is >=1.12.')
#
#     config_name = Path(args.config)
#     if not config_name.exists():
#         logger.error(f'{config_name} not found.')
#
#     cfg = Config.fromfile(args.config)
#     cfg.work_dir = tempfile.TemporaryDirectory().name
#     cfg.log_level = 'WARN'
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#
#     init_default_scope(cfg.get('default_scope', 'mmdet'))
#
#     # TODO: The following usage is temporary and not safe
#     # use hard code to convert mmSyncBN to SyncBN. This is a known
#     # bug in mmengine, mmSyncBN requires a distributed environment，
#     # this question involves models like configs/strong_baselines
#     if hasattr(cfg, 'head_norm_cfg'):
#         cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
#         cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
#             type='SyncBN', requires_grad=True)
#         cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
#             type='SyncBN', requires_grad=True)
#
#     if len(args.shape) == 1:
#         h = w = args.shape[0]
#     elif len(args.shape) == 2:
#         h, w = args.shape
#     else:
#         raise ValueError('invalid input shape')
#     result = {}
#
#     # Supports two ways to calculate flops,
#     # 1. randomly generate a picture
#     # 2. load a picture from the dataset
#     # In two stage detectors, _forward need batch_samples to get
#     # rpn_results_list, then use rpn_results_list to compute flops,
#     # so only the second way is supported
#     try:
#         model = MODELS.build(cfg.model)
#         if torch.cuda.is_available():
#             model.cuda()
#         model = revert_sync_batchnorm(model)
#         data_batch = {'inputs': [torch.rand(3, h, w)], 'batch_samples': [None]}
#         data = model.data_preprocessor(data_batch)
#         result['ori_shape'] = (h, w)
#         result['pad_shape'] = data['inputs'].shape[-2:]
#         model.eval()
#         outputs = get_model_complexity_info(
#             model,
#             None,
#             inputs=data['inputs'],
#             show_table=False,
#             show_arch=True)
#         print(outputs['out_arch'])
#         flops = outputs['flops']
#         params = outputs['params']
#         result['compute_type'] = 'direct: randomly generate a picture'
#
#
#     except TypeError:
#         logger.warning(
#             'Failed to directly get FLOPs, try to get flops with real data')
#         data_loader = Runner.build_dataloader(cfg.val_dataloader)
#         data_batch = next(iter(data_loader))
#         model = MODELS.build(cfg.model)
#         if torch.cuda.is_available():
#             model = model.cuda()
#         model = revert_sync_batchnorm(model)
#         model.eval()
#         _forward = model.forward
#         data = model.data_preprocessor(data_batch)
#         result['ori_shape'] = data['data_samples'][0].ori_shape
#         result['pad_shape'] = data['data_samples'][0].pad_shape
#
#         del data_loader
#         model.forward = partial(_forward, data_samples=data['data_samples'])
#         outputs = get_model_complexity_info(
#             model,
#             None,
#             inputs=data['inputs'],
#             show_table=False,
#             show_arch=False)
#         flops = outputs['flops']
#         params = outputs['params']
#         result['compute_type'] = 'dataloader: load a picture from the dataset'
#
#     flops = _format_size(flops)
#     params = _format_size(params)
#     result['flops'] = flops
#     result['params'] = params
#
#     return result
#
#
# def main():
#     args = parse_args()
#     logger = MMLogger.get_instance(name='MMLogger')
#     result = inference(args, logger)
#     split_line = '=' * 30
#     ori_shape = result['ori_shape']
#     pad_shape = result['pad_shape']
#     flops = result['flops']
#     params = result['params']
#     compute_type = result['compute_type']
#
#     if pad_shape != ori_shape:
#         print(f'{split_line}\nUse size divisor set input shape '
#               f'from {ori_shape} to {pad_shape}')
#     print(f'{split_line}\nCompute type: {compute_type}\n'
#           f'Input shape: {pad_shape}\nFlops: {flops}\n'
#           f'Params: {params}\n{split_line}')
#     print('!!! You should add the flops of dcn v3 manually.')
#     print('!!! The flops of dcn v3 is 5 * input_channels * hight * width * kernel_size * kernel_size.')
#     print('!!! Please be cautious if you use the results in papers. '
#           'You may need to check if all ops are supported and verify '
#           'that the flops computation is correct.')
#
#
# if __name__ == '__main__':
#     main()



# import argparse
# import tempfile
# from functools import partial
# from pathlib import Path
#
# import numpy as np
# import torch
# from mmengine.config import Config, DictAction
# from mmengine.logging import MMLogger
# from mmengine.model import revert_sync_batchnorm
# from mmengine.registry import init_default_scope
# from mmengine.runner import Runner
# from mmengine.utils import digit_version
#
# from mmdet.registry import MODELS
#
# try:
#     from mmengine.analysis import get_model_complexity_info
#     from mmengine.analysis.print_helper import _format_size
# except ImportError:
#     raise ImportError('Please upgrade mmengine >= 0.6.0')
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Get a detector flops')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument(
#         '--num-images',
#         type=int,
#         default=1,
#         help='num images of calculate model flops')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     args = parser.parse_args()
#     return args
#
#
# def inference(args, logger):
#     if digit_version(torch.__version__) < digit_version('1.12'):
#         logger.warning(
#             'Some config files, such as configs/yolact and configs/detectors,'
#             'may have compatibility issues with torch.jit when torch<1.12. '
#             'If you want to calculate flops for these models, '
#             'please make sure your pytorch version is >=1.12.')
#
#     config_name = Path(args.config)
#     if not config_name.exists():
#         logger.error(f'{config_name} not found.')
#
#     cfg = Config.fromfile(args.config)
#     cfg.val_dataloader.batch_size = 1
#     cfg.work_dir = tempfile.TemporaryDirectory().name
#
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#
#     init_default_scope(cfg.get('default_scope', 'mmdet'))
#
#     # TODO: The following usage is temporary and not safe
#     # use hard code to convert mmSyncBN to SyncBN. This is a known
#     # bug in mmengine, mmSyncBN requires a distributed environment，
#     # this question involves models like configs/strong_baselines
#     if hasattr(cfg, 'head_norm_cfg'):
#         cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
#         cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
#             type='SyncBN', requires_grad=True)
#         cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
#             type='SyncBN', requires_grad=True)
#
#     result = {}
#     avg_flops = []
#     data_loader = Runner.build_dataloader(cfg.val_dataloader)
#     model = MODELS.build(cfg.model)
#     if torch.cuda.is_available():
#         model = model.cuda()
#     model = revert_sync_batchnorm(model)
#     model.eval()
#     _forward = model.forward
#
#     for idx, data_batch in enumerate(data_loader):
#         if idx == args.num_images:
#             break
#         data = model.data_preprocessor(data_batch)
#         result['ori_shape'] = data['data_samples'][0].ori_shape
#         result['pad_shape'] = data['data_samples'][0].pad_shape
#         if hasattr(data['data_samples'][0], 'batch_input_shape'):
#             result['pad_shape'] = data['data_samples'][0].batch_input_shape
#         model.forward = partial(_forward, data_samples=data['data_samples'])
#         outputs = get_model_complexity_info(
#             model,
#             None,
#             inputs=data['inputs'],
#             show_table=False,
#             show_arch=False)
#         avg_flops.append(outputs['flops'])
#         params = outputs['params']
#         result['compute_type'] = 'dataloader: load a picture from the dataset'
#     del data_loader
#
#     mean_flops = _format_size(int(np.average(avg_flops)))
#     params = _format_size(params)
#     result['flops'] = mean_flops
#     result['params'] = params
#
#     return result
#
#
# def main():
#     args = parse_args()
#     logger = MMLogger.get_instance(name='MMLogger')
#     result = inference(args, logger)
#     split_line = '=' * 30
#     ori_shape = result['ori_shape']
#     pad_shape = result['pad_shape']
#     flops = result['flops']
#     params = result['params']
#     compute_type = result['compute_type']
#
#     if pad_shape != ori_shape:
#         print(f'{split_line}\nUse size divisor set input shape '
#               f'from {ori_shape} to {pad_shape}')
#     print(f'{split_line}\nCompute type: {compute_type}\n'
#           f'Input shape: {pad_shape}\nFlops: {flops}\n'
#           f'Params: {params}\n{split_line}')
#     print('!!!Please be cautious if you use the results in papers. '
#           'You may need to check if all ops are supported and verify '
#           'that the flops computation is correct.')
#
#
# if __name__ == '__main__':
#     main()


import time  # 新增时间模块
import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')

import sys
import os
from mmengine.registry import MODELS

# 添加自定义模块的路径
sys.path.append('/home/songwei/Share/glj/UNITMODULE/UnitModule/unitmodule/models/detectors')

# 导入 UnitYOLODetector
from unit_detectors import UnitYOLODetector

# 检查是否已经注册，若没有则注册
if 'UnitYOLODetector' not in MODELS._module_dict:
    MODELS.register_module()(UnitYOLODetector)

sys.path.append('/home/songwei/Share/glj/UNITMODULE/UnitModule/unitmodule/models/data_preprocessors')
from data_preprocessor import UnitYOLOv5DetDataPreprocessor

# 检查是否已经注册，若没有则注册
if 'UnitYOLOv5DetDataPreprocessor' not in MODELS._module_dict:
    MODELS.register_module()(UnitYOLOv5DetDataPreprocessor)

sys.path.append('/home/songwei/Share/glj/UNITMODULE/UnitModule/unitmodule/models/data_preprocessors')
from unit_module import UnitModule

# 检查是否已经注册，若没有则注册
if 'UnitModule' not in MODELS._module_dict:
    MODELS.register_module()(UnitModule)

sys.path.append('/home/songwei/Share/glj/UNITMODULE/UnitModule/unitmodule/models/losses')
from transmission_loss import TransmissionLoss
from assisting_color_cast_loss import AssistingColorCastLoss
from color_cast_loss import ColorCastLoss
from saturated_pixel_loss import SaturatedPixelLoss
from total_variation_loss import TotalVariationLoss

# 检查是否已经注册，若没有则注册
if 'TransmissionLoss' not in MODELS._module_dict:
    MODELS.register_module()(TransmissionLoss)
if 'AssistingColorCastLoss' not in MODELS._module_dict:
    MODELS.register_module()(AssistingColorCastLoss)
if 'ColorCastLoss' not in MODELS._module_dict:
    MODELS.register_module()(ColorCastLoss)
if 'SaturatedPixelLoss' not in MODELS._module_dict:
    MODELS.register_module()(SaturatedPixelLoss)
if 'TotalVariationLoss' not in MODELS._module_dict:
    MODELS.register_module()(TotalVariationLoss)


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=1,
        help='num images of calculate model flops')
    # 新增FPS计算参数
    parser.add_argument(
        '--fps-runs',
        type=int,
        default=200,
        help='number of runs for FPS calculation')
    parser.add_argument(
        '--fps-warmup',
        type=int,
        default=20,
        help='warmup iterations before timing')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input shape for FPS calculation (h w)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')
    
    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')
    
    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
    
    result = {}
    avg_flops = []
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    _forward = model.forward
    
    for idx, data_batch in enumerate(data_loader):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        result['ori_shape'] = data['data_samples'][0].ori_shape
        result['pad_shape'] = data['data_samples'][0].pad_shape
        if hasattr(data['data_samples'][0], 'batch_input_shape'):
            result['pad_shape'] = data['data_samples'][0].batch_input_shape
        model.forward = partial(_forward, data_samples=data['data_samples'])
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        avg_flops.append(outputs['flops'])
        params = outputs['params']
        result['compute_type'] = 'dataloader: load a picture from the dataset'
    del data_loader
    
    mean_flops = _format_size(int(np.average(avg_flops)))
    params = _format_size(params)
    result['flops'] = mean_flops
    result['params'] = params
    
    # 新增FPS计算函数
    def calculate_fps(model, input_shape, warmup=20, runs=200):
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 生成随机输入数据
        h, w = input_shape
        tensor = torch.randn(1, 3, h, w)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # 预热阶段
        logger.info(f'FPS warmup ({warmup} runs)')
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(tensor)
        
        # 正式计时
        logger.info(f'FPS measurement ({runs} runs)')
        timings = []
        with torch.no_grad():
            for _ in range(runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()
                timings.append(end - start)
        
        # 计算统计量
        avg_time = np.mean(timings)
        fps = 1.0 / avg_time
        return fps, avg_time
    
    # 在结果字典中添加FPS指标
    result['fps'], result['inference_time'] = calculate_fps(
        model,
        args.input_shape,
        warmup=args.fps_warmup,
        runs=args.fps_runs
    )
    
    return result
    
    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']
    
    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')
    # 新增FPS输出
    fps = result['fps']
    infer_time = result['inference_time']
    print(f'FPS: {fps:.2f} ({infer_time * 1000:.2f} ms per frame)')
    
    print('!!!Note: FPS is calculated with random input tensor '
          f'shape [1,3,{args.input_shape[0]},{args.input_shape[1]}] '
          'and excludes data loading time.')


if __name__ == '__main__':
    main()
