CUDA_VISIBLE_DEVICES=4 nohup python train.py --model espnet --backbone resnet101 --dataset citys --lr 0.0001 --epochs 50 --workers=0 > output2.log
