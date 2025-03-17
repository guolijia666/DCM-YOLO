custom_imports = dict(
    imports=['custom_modules.COCOMetric'],  # 这里添加你的自定义模块路径
    allow_failed_imports=False  # 设置为 True 时，如果导入失败将不会报错
)

#数据集相关信息
data_root = '/home/songwei/Share/glj/UNITMODULE/UnitModule/data/DUOmingan/'

train_img_file = 'images/train'
val_img_file = 'images/test'
train_ann_file = 'annotations/instances_train.json'
val_ann_file = 'annotations/instances_test.json'

# data_root = '/home/songwei/Share/glj/UNITMODULE/UnitModule/enhancedWithRGHS/'
#
# train_img_file = 'images/train'
# val_img_file = 'images/test'
# train_ann_file = 'annotations/instances_train.json'
# val_ann_file = 'annotations/instances_test.json'

# data_root = '/home/songwei/Share/glj/UnitModule/data/URPC2020/'
#
# train_img_file = 'images/train'
# val_img_file = 'images/test'
# train_ann_file = 'annotations/URPC2020_train.json'
# val_ann_file = 'annotations/URPC2020_test.json'

# data_root = '/home/songwei/Share/glj/UNITMODULE/UnitModule/data/COCO 2017/'
#
# train_img_file = 'images/train2017'
# val_img_file = 'images/val2017'
# train_ann_file = 'annotations/instances_train2017.json'
# val_ann_file = 'annotations/instances_val2017.json'

# 这两组值分别表示图像在 BGR 和 RGB 颜色空间下的均值和标准差。它们通常用于图像归一化，以帮助模型更快地收敛。
mean_bgr = [85.603, 148.034, 64.697]
std_bgr = [32.28, 39.201, 26.55]
mean_rgb = [64.697, 148.034, 85.603]
std_rgb = [26.55, 39.201, 32.28]

# 定义了训练和测试集中出现的四个目标类别：海参、海胆、扇贝和海星。
classes = ('holothurian', 'echinus', 'scallop', 'starfish')

# classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite',
#     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut',
#     'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#     'scissors',
#     'teddy bear', 'hair drier', 'toothbrush')

# 图像尺寸和数据集类型
img_scale = (640, 640)
# img_scale = (320, 320)
dataset_type = 'YOLOv5CocoDataset'
evaluator_type = 'mmdet.CocoMetric'   #使用 COCO 风格的评价指标进行评估
# evaluator_type = 'CoCoMetric'

# 训练数据预处理管道：
train_pipeline = [
    dict(type='LoadImageFromFile'),   #从文件中加载图像
    dict(type='mmdet.LoadAnnotations', with_bbox=True),  #加载图像的边界框标注数据
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),  #将图像调整到 img_scale 大小，同时保持纵横比不变
    dict(type='mmdet.Pad',
         pad_to_square=True,
         pad_val=dict(img=(114.0, 114.0, 114.0))),  #将图像填充为正方形，填充值为 114.0（填充后多余的部分会用 (114, 114, 114) 填充）
    dict(type='mmdet.RandomFlip', prob=0.5),  #以 50% 的概率随机水平翻转图像
    dict(type='mmdet.PackDetInputs')  #将图像和标签打包成训练所需的输入格式
]
# 测试数据预处理管道:与训练管道类似，只是没有随机翻转等数据增强操作。此外，meta_keys 保存了一些与图像相关的元数据信息
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad',
         pad_to_square=True,
         pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

num_gpu = 1  #训练过程中的 GPU 数量
# train_bs = 1  #每个 GPU 上的训练批次大小
train_bs = 8
# val_bs = 1  #验证阶段的每个批次大小为 1
val_bs = 1
#这是自动学习率调整配置,enable=False：表示不启用自动学习率调整功能。如果启用（enable=True），学习率会根据批次大小自动进行缩放。
auto_scale_lr = dict(enable=False, base_batch_size=train_bs * num_gpu)   #base_batch_size这是用于学习率缩放的基础批次大小
# 训练数据加载器：
train_dataloader = dict(
    batch_size=train_bs,  #每个 GPU 上的批处理大小为 4
    num_workers=train_bs,  #为数据加载分配了 4 个工作进程，用于并行加载数据。
    persistent_workers=True,  #启用后，加载器的工作进程将一直保持活跃，不会每个 epoch 重启，提升数据加载效率。
    collate_fn=dict(type='yolov5_collate'),  #自定义的函数 yolov5_collate，用于将多个样本组合成一个批次
    sampler=dict(type='DefaultSampler', shuffle=True), #DefaultSampler 用于打乱数据顺序
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'), #mmdet.AspectRatioBatchSampler 通过考虑图像的长宽比来优化批处理
    dataset=dict(  #定义了数据集的具体结构，包括图像路径、标注文件、类别、过滤配置（如过滤掉没有标注或尺寸过小的图像）和数据处理管道
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_file),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ))

# 验证数据加载器:
# 验证数据加载器与训练的差异：
# batch_size：验证集每个批次的大小为 1。
# shuffle=False：不打乱验证集数据。
# drop_last=False：不会丢弃最后不足一个批次的数据。
# test_mode=True：验证数据加载器在测试模式下使用，不加载标签等信息
val_dataloader = dict(
    batch_size=val_bs,
    num_workers=val_bs * 2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_file),
        test_mode=True,
        pipeline=test_pipeline,
    ))

test_dataloader = val_dataloader
# 评估器：
# type：使用 COCO 格式的评价器（mmdet.CocoMetric），计算 bbox（边界框）的检测性能。
# ann_file：评估时使用验证集的标注文件。
# format_only=False：表示不仅仅进行格式转换，还要计算实际的评估指标。
val_evaluator = dict(
    type=evaluator_type,
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False)
    # format_only=True,
    # outfile_prefix='./work_dirs/PR/DUO/YOLOv5')
test_evaluator = val_evaluator
