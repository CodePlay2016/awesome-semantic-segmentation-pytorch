CUDA_VISIBLE_DEVICES=3 nohup python train.py --model deeplabv3 --backbone resnet101 --dataset citys --lr 0.0001 --epochs 50 --workers 0 --save-dir ../model > output.log
