# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from unitmodule.models.detectors import register_unit_distributed

#主要用于启动模型的训练。它提供了从命令行解析参数、加载配置文件、处理自动混合精度训练、自动学习率调整等功能
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path') #config，表示训练配置文件的路径
    parser.add_argument('--work-dir', help='the dir to save logs and models') #--work-dir，指定保存日志和模型的工作目录
    parser.add_argument( # --amp 参数，启用自动混合精度训练（AMP）。action='store_true' 表示如果该参数存在，则 amp=True，否则为 False。
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument( #添加 --auto-scale-lr 参数，启用自动缩放学习率功能
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument( #添加 --resume 参数。可以指定一个检查点路径用于从中恢复训练；如果没有指定检查点路径，脚本会尝试自动从工作目录中的最新检查点恢复
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(  #--cfg-options 参数，用于在命令行中覆盖配置文件中的一些设置。格式为 key=value
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(  #--launcher 参数，用于选择分布式训练的启动方式，如 pytorch、slurm、mpi 等
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0) #--local_rank 参数，用于分布式训练时指定当前 GPU 的编号。这个参数由 torch.distributed.launch 传入
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args() #调用 parse_args 函数解析命令行参数

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()  #调用一个函数 setup_cache_size_limit_of_dynamo()，这个函数可能用于减少重复编译并提高训练速度

    # 使用 Config.fromfile 函数从指定的配置文件中加载配置。然后将解析得到的 launcher 参数保存到 cfg 中
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None: #如果命令行中提供了 --cfg-options 参数，使用 merge_from_dict 将这些配置覆盖到加载的配置文件中
        cfg.merge_from_dict(args.cfg_options)

    # --------------------------------------------------------
    # dynamic import customs modules
    # import modules from import_dir as a/b/c/ dir, registry will be updated
    # 动态导入自定义模块
    if hasattr(cfg, 'import_dir'):
        import importlib

        import_dir = cfg.import_dir
        module_path = import_dir.replace('/', '.')
        import_lib = importlib.import_module(module_path)

    # dynamic import for ddp of UnitModule if key with_unit_module is True
    register_unit_distributed(cfg)
    # --------------------------------------------------------

    # work_dir is determined in this priority: CLI > segment in file > filename
    # 确定工作目录的优先级：
    # 如果命令行指定了 - -work - dir，则使用它。
    # 如果配置文件中没有
    # work_dir，则根据配置文件的名称自动生成一个目录，放在. / work_dirs / 下。
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training  启用自动混合精度训练
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR  自动调整学习率
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume 恢复训练
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config  构建 Runner 并开始训练
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()  #启动训练流程


if __name__ == '__main__':
    main()
