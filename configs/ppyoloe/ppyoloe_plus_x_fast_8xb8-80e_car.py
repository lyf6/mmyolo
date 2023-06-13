_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/ppyoloe_plus_x_obj365_pretrained-43a8000d.pth'  # noqa

deepen_factor = 1.33
widen_factor = 1.25


data_root = 'data/uavcar/JPEGImages'
class_name = ('car', )
dataset_type='YOLOv5CocoDataset'
# parameters that often need to be modified
img_scale = (640, 640)  # width, height
max_epochs = 80
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])
save_epoch_intervals = 5
train_batch_size_per_gpu = 1
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

base_lr = 0.0004


model = dict(
    data_preprocessor=dict(
    # use this to support multi_scale training
   
    batch_augments=[
        dict(
            type='PPYOLOEBatchRandomResize',
            random_size_range=(320, 800))
    ]),
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor,
                                    num_classes=num_classes)),
    train_cfg=dict(
        initial_epoch=30,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=13,
            alpha=1,
            beta=6,
            eps=1e-9)
            ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=1000,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300)
    )

# train_pipeline = [
#     dict(type='PPYOLOERandomCrop', scaling=[0.8, 1])
# ]
# train_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='PPYOLOERandomDistort'),
#     dict(type='mmdet.Expand', mean=(103.53, 116.28, 123.675)),
#     dict(type='PPYOLOERandomCrop',scaling=[0.8, 1]),
#     dict(type='mmdet.RandomFlip', prob=0.5),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
#                    'flip_direction'))
# ]


test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='mmdet.FixShapeResize',
        width=img_scale[0],
        height=img_scale[1],
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        ann_file='data/uavcar/dataset.json',
        data_prefix=dict(img='data/uavcar/car'))
    )

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        test_mode=True,
        data_prefix=dict(img='data/uavcar/JPEGImages'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file='result.json',
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file = data_root+'result.json',
    metric='bbox')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals))

train_cfg = dict(
    val_interval=save_epoch_intervals)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False),
    paramwise_cfg=dict(norm_decay_mult=0.))