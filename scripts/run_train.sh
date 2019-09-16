CUDA_VISIBLE_DEVICES=5 nohup python train.py --model deeplabv3 --backbone resnet152 --dataset citys --lr 0.0001 --epochs 200 --workers 0 --save-dir ../model > output3.log
