custom_imports = dict(
    imports=['custom_modules.COCOMetric'],  
    allow_failed_imports=False  
)


data_root = '/home/songwei/Share/glj/UNITMODULE/UnitModule/data/DUO/'

train_img_file = 'images/train'
val_img_file = 'images/test'
train_ann_file = 'annotations/instances_train.json'
val_ann_file = 'annotations/instances_test.json'


mean_bgr = [85.603, 148.034, 64.697]
std_bgr = [32.28, 39.201, 26.55]
mean_rgb = [64.697, 148.034, 85.603]
std_rgb = [26.55, 39.201, 32.28]

# 定义了训练和测试集中出现的四个目标类别：海参、海胆、扇贝和海星。
classes = ('holothurian', 'echinus', 'scallop', 'starfish')


img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
evaluator_type = 'mmdet.CocoMetric'   #使用 COCO 风格的评价指标进行评估
# evaluator_type = 'CoCoMetric'


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
train_bs = 8
val_bs = 1

auto_scale_lr = dict(enable=False, base_batch_size=train_bs * num_gpu)   #base_batch_size这是用于学习率缩放的基础批次大小

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

val_evaluator = dict(
    type=evaluator_type,
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False)
    # format_only=True,
    # outfile_prefix='./work_dirs/PR/DUO/YOLOv5')
test_evaluator = val_evaluator
