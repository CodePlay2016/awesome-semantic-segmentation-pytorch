CUDA_VISIBLE_DEVICES=3 nohup python train.py --model espnet --backbone resnet101 --dataset citys --lr 0.0001 --epochs 500 --workers=0 > output4.log
