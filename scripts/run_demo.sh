# python demo.py --model deeplabv3_resnet101_citys --dataset citys --input-pic /home/hufq/data/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000192_000019_leftImg8bit.png --out-dir /home/hufq/eval
CUDA_VISIBLE_DEVICES=6 python demo.py \
    --model deeplabv3_resnet152_mapillary \
    --save-folder ../model/mapillary \
    --dataset mapillary \
    --input-pic /home/hufq/test_changyang.png \
    --out-dir /home/hufq/eval \
    --repetition 1
# --demo-dir "/home/hufq/LOD/fish_eye_correct/*.png" \
# for img in '/home/hufq/data/apollo_selected'/*5.jpg
# do
#     python demo.py --model deeplabv3_resnet101_citys --dataset citys --input-pic $img --out-dir /home/hufq/eval
# done
