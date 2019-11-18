# python demo.py --model deeplabv3_resnet101_citys --dataset citys --input-pic /home/hufq/data/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000192_000019_leftImg8bit.png --out-dir /home/hufq/eval
CUDA_VISIBLE_DEVICES=0 python demo.py \
    --model deeplabv3_resnet50_mapillary \
    --save-folder ../model/mapillary_selected \
    --dataset mapillary \
    --demo-dir "/home/hufq/data/correct_cylinder/*.png" \
    --input-size '720,1280' \
    --input-pic /home/hufq/test_changyang.png \
    --out-dir /home/hufq/data/cylinder/out_720_selected \
    --repetition 1
# for img in '/home/hufq/data/apollo_selected'/*5.jpg
# do
#     python demo.py --model deeplabv3_resnet101_citys --dataset citys --input-pic $img --out-dir /home/hufq/eval
# done
