# 主要用于控制训练过程中各种环境设置、日志记录、可视化以及一些通用的钩子（hooks）操作

default_scope = 'mmyolo' 
log_level = 'INFO'  
load_from = None  
resume = False  
# 环境配置 (env_cfg)：
env_cfg = dict(
    cudnn_benchmark=False,  
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(seed=None)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'), 
    logger=dict(type='LoggerHook', interval=10), 
    param_scheduler=dict(type='ParamSchedulerHook'), 
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'), 
    visualization=dict(type='mmdet.DetVisualizationHook')) 
