# python demo.py --model deeplabv3_resnet101_citys --dataset citys --input-pic /home/hufq/data/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000192_000019_leftImg8bit.png --out-dir /home/hufq/eval
CUDA_VISIBLE_DEVICE=2 python demo.py --model deeplabv3_resnet101_mapillary --save-folder ../model --dataset mapillary --input-pic ../datasets/mapillary/testing/images/01C9HdKM26NXzPEnlzH2Xw.jpg --out-dir /home/hufq/eval --repetition 1
# for img in '/home/hufq/data/apollo_selected'/*5.jpg
# do
#     python demo.py --model deeplabv3_resnet101_citys --dataset citys --input-pic $img --out-dir /home/hufq/eval
# done
