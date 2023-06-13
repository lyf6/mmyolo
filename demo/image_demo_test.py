# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path
import mmengine
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from large_image_demo_test import json

def parse_args():
    parser = ArgumentParser()
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
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--class-name',
        nargs='+',
        type=str,
        help='Only Save those classes if set')
    parser.add_argument(
        '--to-labelme',
        action='store_true',
        help='Output labelme style label file')
    parser.add_argument(
        '--max_batch_num_pred',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')
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

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # get model class name
    dataset_classes = model.dataset_meta.get('classes')

    # ready for labelme format if it is needed
    to_label_format = LabelmeFormat(classes=dataset_classes)

    # check class name
    if args.class_name is not None:
        for class_name in args.class_name:
            if class_name in dataset_classes:
                continue
            show_data_classes(dataset_classes)
            raise RuntimeError(
                'Expected args.class_name to be one of the list, '
                f'but got "{class_name}"')

    # start detector inference
    progress_bar = ProgressBar(len(files))
    sam_checkpoint = "/home/yf/Documents/segment-anything/weights/vit_h.pth"
    model_type = "vit_h"

    device = args.device

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    anns = []
    ann_images = []
    for img_id, file in enumerate(files):
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        
        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        progress_bar.update()

        # Get candidate predict info with score threshold
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]
        
        bboexes = pred_instances.bboxes.clone()

        if(len(bboexes)>0):
            num_pred = len(bboexes)
            N = args.max_batch_num_pred
            num_batches = int(np.ceil(num_pred / N))
            sam_img = mmcv.imconvert(img, 'bgr', 'rgb')
            
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
            pred_instances.masks = torch.stack(masks, dim=0)        
            width, height =   result.metainfo['ori_shape'][1], result.metainfo['ori_shape'][0]
            ann_image, ann = json(filename, pred_instances, img_id+1, width, height)
            if(ann_image is not None):
                ann_images.append(ann_image)
                anns.extend(ann)

        if args.to_labelme:
            # save result to labelme files
            out_file = out_file.replace(
                os.path.splitext(out_file)[-1], '.json')
            to_label_format(pred_instances, result.metainfo, out_file,
                            args.class_name)
            continue

        # visualizer.add_datasample(
        #     filename,
        #     img,
        #     data_sample=result,
        #     draw_gt=False,
        #     show=args.show,
        #     wait_time=0,
        #     out_file=out_file,
        #     pred_score_thr=args.score_thr)

    if not args.show and not args.to_labelme:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')

    elif args.to_labelme:
        print_log('\nLabelme format label files '
                  f'had all been saved in {args.out_dir}')

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
            'id': 0,
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
    out_json_path = "./mulresult_det.json"
    mmengine.dump(results, out_json_path)

if __name__ == '__main__':
    main()
