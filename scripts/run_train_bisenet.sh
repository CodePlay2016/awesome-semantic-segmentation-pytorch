CUDA_VISIBLE_DEVICES=2 \
nohup python train.py \
    --model bisenet \
    --backbone resnet18 \
    --batch-size 64 \
    --dataset mapillary \
    --lr 0.01 \
    --epochs 200 \
    --workers 0 \
    --multi-cuda \
    --gpu-ids 1 \
    --save-dir ../model/mapillary_test/ > bisenet_mapi_selected.log &
#CUDA_VISIBLE_DEVICES=5 python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model
