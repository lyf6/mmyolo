_base_ = './yolov8_l_syncbn_fast_8xb16-500e_coco.py'
data_root=''
train_ann_file='train.json'
train_data_prefix='JPEGImages_train'
class_name = ('car', )
base_lr = 0.0001
img_scale = (640, 640)
max_epochs = 50  # Maximum training epochs
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

test_dataloader = val_dataloader

deepen_factor = 1.00
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        head_module=dict
        (widen_factor=widen_factor,
        num_classes=num_classes)
        ),
    train_cfg=dict(
    assigner=dict(
        type='BatchTaskAlignedAssigner', 
    num_classes=num_classes)
        )
        )
train_num_workers = 2
load_from = 'pretrained/yolov8_x/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'
# test_pipeline = [
#     dict(type='data/uavcar/car', file_client_args=_base_.file_client_args),
#     dict(
#         type='mmdet.Resize',
#         scale=img_scale,
#         allow_scale_up=False,
#         pad_val=dict(img=114)),
#     dict(type='/home/yf/Documents/mmyolo/data/uavcar/Annotations', with_bbox=True, _scope_='mmdet'),
#     dict(
#         type='mmdet.PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'pad_param'))
#     ]
# test_dataloader = dict(
#    batch_size=1,
#    num_workers=2,
#    persistent_workers=True,
#    drop_last=False,
#    sampler=dict(type='DefaulstSampler', shuffle=False),
#    dataset=dict(
#        type='YOLOv5CocoDataset',
#        data_root=data_root,
#        ann_file=data_root + '/home/yf/Documents/mmyolo/data/uavcar/dataset.json',
#        data_prefix=dict(img='data/uavcar/JPEGImages'),
#        test_mode=True,
#        pipeline=test_pipeline))

# test_evaluator = dict(
#    type='mmdet.Cocometric',
#    ann_file=data_root + '/home/yf/Documents/mmyolo/data/uavcar/dataset.json',
#    metric='bbox',
#    format_only=True,
#    outfile_prefix='./work_dirs/car_detection/test'
# )


# #Visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
# #dict(type='TensorboardVisBackend')])

