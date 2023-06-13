import fiftyone as fo
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import fiftyone.utils.coco as fouc

img_path = 'data/farmland/dataset/test/greenhouse_2019'
#"/home/yf/disk/building_competection/track1/train/rgb/"
labels_path= 'data/farmland/dataset/test/instances_greenhouse_test2019.json'
pred_json="mulresult.json"
coco = COCO(annotation_file=pred_json)
#"png.json"
#"/home/yf/disk/building_competection/track1/roof_fine_train.json"
# The type of the dataset being imported
classes=[" ", "farmland"]
def getid_byfilename(coco, file_name):
    for img in coco.imgs:
        if(file_name==coco.imgs[img]['file_name']):
            return img, coco.imgs[img]['width'], coco.imgs[img]['height']
    return None, None, None



dataset_type = fo.types.COCODetectionDataset  # for example
dataset = fo.Dataset.from_dir(
    data_path=img_path,
    labels_path=labels_path,
    dataset_type=dataset_type,
    # dynamic=True,
    include_id=True
)

fouc.add_coco_labels(dataset, "predictions", pred_json, classes, label_type="detections")


# id = 0
# for sample in dataset:
#     detections = []
#     img_id, width, height = getid_byfilename(coco, sample.filename)
#     if img_id is not None:
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         targets = coco.loadAnns(ann_ids)
#         for target in targets:
#             label = target["category_id"]  # get object class id
#             polygons = target["segmentation"]   # get object polygons
#             x, y, w, h = target["bbox"]
#             rel_box = [x / width, y / height, w / width, h / height]
#             # rel_box = [x, y, w, h]
#             score = target["score"]
#             mask = coco.annToMask(target)
#             if len(mask.shape) < 3:
#                 mask = mask[..., None]
#             mask = mask.any(axis=2)
            
#             detections.append(
#                 fo.Detection(
#                     label=classes[label-1],
#                     bounding_box=rel_box,
#                     confidence=float(score),
#                     mask = mask
#                 )
#             )
#     sample["predictions"] = fo.Detections(detections=detections)
#     sample.save()
        

        # print(sample)
print(dataset.count("predictions"))

if __name__ == "__main__":
    session = fo.launch_app(dataset)
    while True:
        try:
            pass
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt exception, closing...")
            session.close()
            break   