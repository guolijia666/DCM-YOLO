# 主要用于控制训练过程中各种环境设置、日志记录、可视化以及一些通用的钩子（hooks）操作

default_scope = 'mmyolo'  #default_scope = 'mmyolo'：指定默认的配置范围或项目名称，这里设置为 'mmyolo'，表明这是在使用 MMYOLO 框架。
log_level = 'INFO'  #定义日志记录的级别。INFO 级别意味着会记录训练过程中的常规信息，类似于中等详细程度的日志输出（不输出调试信息）。
load_from = None  #表示不从预训练模型中加载权重。如果提供路径，训练会从指定的模型权重开始加载
resume = False  #表示不恢复上一次中断的训练。如果设为 True，则可以从上次训练中断的地方继续训练
# 环境配置 (env_cfg)：
env_cfg = dict(
    cudnn_benchmark=False, #这个参数用于控制是否启用 cudnn 的自动优化（benchmarking）。通常，启用 True 会在特定硬件上优化卷积操作，但在不同尺寸的输入下，启用可能反而导致效率问题，因此这里关闭
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# 控制训练中的随机数种子。如果设定一个特定的种子，训练过程中的数据顺序、初始化权重等会固定，确保实验结果的可重复性。这里设为 None，表示不固定随机种子，每次运行可能会有不同的随机结果。
randomness = dict(seed=None)

#可视化设置 (vis_backends 和 visualizer)：
# vis_backends：定义了两种可视化后端：
# LocalVisBackend：本地可视化后端，可能直接输出图像或结果到本地文件系统。
# TensorboardVisBackend：使用 TensorBoard 进行可视化，可以在 TensorBoard 中查看训练过程中的损失、指标等图表。
# visualizer：这是一个通用的可视化器，类型为 mmdet.DetLocalVisualizer，主要用于检测任务。它整合了上述的可视化后端，用于在训练或推理过程中进行图像、结果的可视化展示。
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# 日志处理器 (log_processor)：window_size = 50：设定日志的窗口大小，每 50 个 iteration（迭代）汇总输出一次日志信息。
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
# 默认的钩子函数:钩子（hook）是深度学习框架中用于在训练或推理过程中执行某些操作的机制，比如记录日志、保存模型等。
default_hooks = dict(
    timer=dict(type='IterTimerHook'), #记录每个 iteration 的s时间
    logger=dict(type='LoggerHook', interval=10), #控制日志记录的钩子。这里设定每 50 次 iteration 输出一次日志，包含损失、精度等信息。
    param_scheduler=dict(type='ParamSchedulerHook'), #用于控制学习率、动量等参数的调整策略。具体的参数调整策略可以在训练配置的其他部分定义。
    #负责在训练过程中保存模型的钩子：
    # interval=1：表示每个 epoch 保存一次模型。
    # max_keep_ckpts=3：最多保留 3 个最新的 checkpoint，旧的会被删除。
    # save_best='coco/bbox_mAP'：根据 COCO 数据集上的 bbox_mAP（边界框平均精度）保存最好的模型。
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'), #用于在分布式训练中保证每个进程使用不同的随机种子，以避免数据重复
    visualization=dict(type='mmdet.DetVisualizationHook')) #用于在训练过程中进行检测结果可视化的钩子，会调用上面配置的 visualizer 来展示检测框或其他输出。
