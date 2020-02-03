CUDA_VISIBLE_DEVICES=6 \
nohup python train.py \
    --model deeplabv3 \
    --backbone mobilenetv2 \
    --batch-size 24 \
    --dataset mapillary \
    --lr 1e-2 \
    --val-epoch 0.5 \
    --epochs 200 \
    --workers 0 \
    --save-dir ../model/mapillary_test_1/ > ../result/output_dlv3_mbv2_mapi_selected_gpu6.log &
#CUDA_VISIBLE_DEVICES=5 python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model
