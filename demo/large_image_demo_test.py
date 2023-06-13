# Copyright (c) OpenMMLab. All rights reserved.
"""Perform MMYOLO inference on large images (as satellite imagery) as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth # noqa: E501, E261.

python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
    checkpoint/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
```
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path
import mmengine
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
import cv2
import torch
from mmengine.structures import InstanceData
try:
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError('Please run "pip install -U sahi" '
                      'to install sahi first for large image inference.')
import matplotlib.pyplot as plt
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.large_image import merge_results_by_nms, shift_predictions
from mmyolo.utils.misc import get_file_list
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.mask import encode

def parse_args():
    parser = ArgumentParser(
        description='Perform MMYOLO inference on large images.')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument(
        '--patch-size', type=int, default=2048, help='The size of patches')
    parser.add_argument(
        '--patch-overlap-ratio',
        type=float,
        default=0.4,
        help='Ratio of overlap between two patches')
    parser.add_argument(
        '--merge-iou-thr',
        type=float,
        default=0.2,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--merge-nms-type',
        type=str,
        default='nms',
        help='NMS type for merging results')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Export debug results before merging')
    parser.add_argument(
        '--save-patch',
        action='store_true',
        help='Save the results of each patch. '
        'The `--debug` must be enabled.')
    parser.add_argument(
        '--max_batch_num_pred',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
    
    args = parser.parse_args()
    return args


def single_encode(x):
    rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def json(img_name, result, img_id, width, height):
    image = dict()
    
    image['file_name'] = img_name
    image['width'] = width
    image['height'] = height
    image['id'] = img_id
    
    bboxes = result['bboxes'].cpu().numpy().astype(np.int)
    bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
    scores = result['scores'].cpu().numpy()
    cls_id = result['labels'].cpu().numpy()
    det_num = len(bboxes)
    if(det_num==0):
        return None, None
    masks = result['masks']
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()


    datas = []
    for id in range(det_num):
        data = dict()
        # tmp = (segs[id].reshape(1, -1)*512).astype(np.int).clip(0, 512).tolist()
        data['image_id'] = img_id

        # x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        data['area'] = int(bboxes[id][2]*bboxes[id][3])
        data['bbox'] = bboxes[id].tolist()
        data['score'] = scores[id]
        data['category_id'] = cls_id[id] + 1
        mask = masks[id]
        data['segmentation'] = single_encode(mask)
        data['id'] = int(id)
        datas.append(data)
        id += 1
    return image, datas

# def write_json()



def main():
    args = parse_args()

    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta:
        assert 'tta_model' in config, 'Cannot find ``tta_model`` in config.' \
                                      " Can't use tta !"
        assert 'tta_pipeline' in config, 'Cannot find ``tta_pipeline`` ' \
                                         "in config. Can't use tta !"
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    # TODO: TTA mode will error if cfg_options is not set.
    #  This is an mmdet issue and needs to be fixed later.
    # build the model from a config file and a checkpoint file
    model = init_detector(
        config, args.checkpoint, device=args.device, cfg_options={})

    if args.deploy:
        switch_to_deploy(model)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    print(f'Performing inference on {len(files)} images.... '
          'This may take a while.')
    progress_bar = ProgressBar(len(files))

    # load sam model
    sam_checkpoint = "/home/yf/Documents/segment-anything/weights/vit_h.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    anns = []
    ann_images = []
    for img_id, file in enumerate(files):
        # read image
        img = mmcv.imread(file)

        # arrange slices
        height, width = img.shape[:2]
        sliced_image_object = slice_image(
            img,
            slice_height=args.patch_size,
            slice_width=args.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=args.patch_overlap_ratio,
            overlap_width_ratio=args.patch_overlap_ratio,
        )

        # perform sliced inference
        slice_results = []
        start = 0
        while True:
            # prepare batch slices
            end = min(start + args.batch_size, len(sliced_image_object))
            images = []
            for sliced_image in sliced_image_object.images[start:end]:
                images.append(sliced_image)

            # forward the model
            dets = inference_detector(model, images)
            pred_inst = dets[0].pred_instances
            # masks = torch.zeros((len(pred_inst),images[0].shape[0],images[0].shape[1]), device=device).bool()
            index = pred_inst.scores >= args.score_thr
            
            data_sample = InstanceData(metainfo={})
           
            # pred_inst.masks = masks
            data_sample.bboxes = pred_inst.bboxes[index]
            data_sample.scores = pred_inst.scores[index]
            data_sample.labels = pred_inst.labels[index]
            bboexes = pred_inst.bboxes.clone()[index]
            # pred_inst = pred_inst[]
            if(len(bboexes)>0):
                num_pred = len(bboexes)
                N = args.max_batch_num_pred
                num_batches = int(np.ceil(num_pred / N))
                sam_img = mmcv.imconvert(images[0], 'bgr', 'rgb')
                
                predictor.set_image(sam_img)
                
                masks = []
                for i in range(num_batches):
                    left_index = i * N
                    right_index = (i + 1) * N
                    if i == num_batches - 1:
                        batch_boxes = bboexes[left_index:]
                    else:
                        batch_boxes = bboexes[left_index:right_index]
                    torch.cuda.empty_cache()
                    transformed_boxes = predictor.transform.apply_boxes_torch(batch_boxes, sam_img.shape[:2])
                    
                    batch_masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                    batch_masks = batch_masks.squeeze(1)
                    masks.extend([*batch_masks])
                    # transformed_boxes = []
                    
                data_sample.masks = torch.stack(masks, dim=0)
            # dets[0].mask = ""
            dets[0].pred_instances = data_sample
            slice_results.extend(dets)
            predictor.reset_image()
            # torch.cuda.empty_cache()
            if end >= len(sliced_image_object):
                break
            start += args.batch_size

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)

        img = mmcv.imconvert(img, 'bgr', 'rgb')

        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # export debug images
        if args.debug:
            # export sliced image results
            name, suffix = os.path.splitext(filename)

            shifted_instances = shift_predictions(
                slice_results,
                sliced_image_object.starting_pixels,
                src_image_shape=(height, width))
            merged_result = slice_results[0].clone()
            merged_result.pred_instances = shifted_instances

            debug_file_name = name + '_debug' + suffix
            debug_out_file = None if args.show else os.path.join(
                args.out_dir, debug_file_name)
            visualizer.set_image(img.copy())

            debug_grids = []
            for starting_point in sliced_image_object.starting_pixels:
                start_point_x = starting_point[0]
                start_point_y = starting_point[1]
                end_point_x = start_point_x + args.patch_size
                end_point_y = start_point_y + args.patch_size
                debug_grids.append(
                    [start_point_x, start_point_y, end_point_x, end_point_y])
            debug_grids = np.array(debug_grids)
            debug_grids[:, 0::2] = np.clip(debug_grids[:, 0::2], 1,
                                           img.shape[1] - 1)
            debug_grids[:, 1::2] = np.clip(debug_grids[:, 1::2], 1,
                                           img.shape[0] - 1)

            palette = np.random.randint(0, 256, size=(len(debug_grids), 3))
            palette = [tuple(c) for c in palette]
            line_styles = random.choices(['-', '-.', ':'], k=len(debug_grids))
            visualizer.draw_bboxes(
                debug_grids,
                edge_colors=palette,
                alpha=1,
                line_styles=line_styles)
            visualizer.draw_bboxes(
                debug_grids, face_colors=palette, alpha=0.15)

            visualizer.draw_texts(
                list(range(len(debug_grids))),
                debug_grids[:, :2] + 5,
                colors='w')

            visualizer.add_datasample(
                debug_file_name,
                visualizer.get_image(),
                data_sample=merged_result,
                draw_gt=False,
                show=args.show,
                wait_time=0,
                out_file=debug_out_file,
                pred_score_thr=args.score_thr,
            )

            if args.save_patch:
                debug_patch_out_dir = os.path.join(args.out_dir,
                                                   f'{name}_patch')
                for i, slice_result in enumerate(slice_results):
                    patch_out_file = os.path.join(
                        debug_patch_out_dir,
                        f'{filename}_slice_{i}_result.jpg')
                    image = mmcv.imconvert(sliced_image_object.images[i],
                                           'bgr', 'rgb')

                    visualizer.add_datasample(
                        'patch_result',
                        image,
                        data_sample=slice_result,
                        draw_gt=False,
                        show=False,
                        wait_time=0,
                        out_file=patch_out_file,
                        pred_score_thr=args.score_thr,
                    )

        image_result = merge_results_by_nms(
            slice_results,
            sliced_image_object.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg={
                'type': args.merge_nms_type,
                'iou_threshold': args.merge_iou_thr
            })
        pred_dict =image_result.pred_instances

        # width, height, _ = img.shape
        ann_image, ann = json(filename, pred_dict, img_id, width, height)
        if(ann_image is not None):
            ann_images.append(ann_image)
            anns.extend(ann)

        progress_bar.update()

    if not args.show or (args.debug and args.save_patch):
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')
    
    INFO = {
    "description": "farmland",
    "url": "",
    "version": "0.1.0",
    "year": 2023,
    "contributor": "liuyanfei",
    "date_created": "2023"
    }

    LICENSES = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]
    CATEGORIES = [
        {
            'id': 1,
            'name': 'farmland',
            'supercategory': 'farmland',
        },
    ]
    results = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    results['annotations'] = anns
    results['images'] = ann_images
    out_json_path = "./result.json"
    mmengine.dump(results, out_json_path)

if __name__ == '__main__':
    main()
