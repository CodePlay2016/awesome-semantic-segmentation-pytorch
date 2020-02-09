CUDA_VISIBLE_DEVICES=2,3 \
nohup python train.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --batch-size 64 \
    --dataset mapillary \
    --lr 1e-3 \
    --val-epoch 0.5 \
    --epochs 200 \
    --multi-cuda \
    --gpu-ids 0,1 \
    --workers 4 \
    --save-dir ../model/mapillary_test/ > ../result/output_dlv3_res18_mapi_selected_gpu_0_1.log &
#CUDA_VISIBLE_DEVICES=5 python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model
