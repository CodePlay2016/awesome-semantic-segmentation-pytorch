CUDA_VISIBLE_DEVICES=0,2 \
python train.py \
    --model deeplabv3 \
    --backbone mobilenetv2 \
    --batch-size 24 \
    --dataset mapillary \
    --lr 1e-2 \
    --val-epoch 0.5 \
    --epochs 200 \
    --multi-cuda \
    --gpu-ids 0,1 \
    --workers 4 \
    --save-dir ../model/ #> ../result/output_dlv3_mbv2_mapi_selected_gpu6.log &
#CUDA_VISIBLE_DEVICES=5 python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model
