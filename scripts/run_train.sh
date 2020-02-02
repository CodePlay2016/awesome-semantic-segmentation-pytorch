CUDA_VISIBLE_DEVICES=7 \
nohup python train.py \
    --model deeplabv3 \
    --backbone resnet50 \
    --batch-size 24 \
    --dataset mapillary \
    --lr 1e-4 \
    --epochs 200 \
    --workers 0 \
    --save-dir ../model/mapillary_test/ > ../result/output_dlv3_rn50_mapi_selected_gpu3.log &
#CUDA_VISIBLE_DEVICES=5 python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model
