import fiftyone as fo

# The directory containing the dataset to import
img_path = 'JPEGImages_test'
#"/home/yf/disk/building_competection/track1/train/rgb/"
labels_path= 'test.json'
#"png.json"
#"/home/yf/disk/building_competection/track1/roof_fine_train.json"
# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example
dataset = fo.Dataset.from_dir(
    data_path=img_path,
    labels_path=labels_path,
    dataset_type=dataset_type
)
if __name__ == "__main__":
    session = fo.launch_app(dataset)
    while True:
        try:
            pass
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt exception, closing...")
            session.close()
            break

        # rm -rf ~/.fiftyone/*