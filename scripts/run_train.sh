#CUDA_VISIBLE_DEVICES=5 nohup python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model > output3.log
CUDA_VISIBLE_DEVICES=5 python train.py --model deeplabv3 --backbone resnet101 --dataset mapillary --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model
