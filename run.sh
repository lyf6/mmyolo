export CUDA_VISIBLE_DEVICES=1,2
CONFIG_FILE=configs/yolov8/yolov8_x_syncbn_fast_8xb16-500e_car.py 
           # work_dirs/yolov8_x_syncbn_fast_8xb16-500e_car/epoch_10.pth \
          #  --show-dir show_results
GPU_NUM=2
./tools/dist_train.sh \
${CONFIG_FILE} \
${GPU_NUM} \