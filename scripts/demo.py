import os
import pdb
import sys
import glob
import argparse
import time
import torch
import cv2

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
import numpy as np
from core.utils.visualize import get_color_pallete, get_color_pallete_c
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc','pascal_aug','ade20k','citys','mapillary'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg',
                    help='path to the input picture')
parser.add_argument('--input-size', type=str, default='',
                    help='size of input picture')
parser.add_argument('--demo-dir', type=str, default=None, help="whether to show a directory of images")
parser.add_argument('--make-transparent', action='store_true', default=False)
parser.add_argument('--alpha', type=int, default=0.5)
parser.add_argument('--input-dir', type=str, default='../datasets/mapillary/testing/images',
                    help='path to the input picture')
parser.add_argument('--out-dir', default='./eval', type=str,
                    help='path to save the predict result')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--repetition', type=int, default=1)
args = parser.parse_args()

REP = args.repetition

def demo_vedio(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print('using device %s' % device)
    # output folder
    # load vedio
    read_video_path = "../tests/test_cyl_front.avi"
    video_cap = cv2.VideoCapture(read_video_path)
    write_video_path = '../result/demo.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    size = config.input_size.split(',')
    size = [int(a.strip()) for a in size]
    video_writer = cv2.VideoWriter(write_video_path, fourcc, 30,
                                    (size[1], size[0]))
    
    
    # load model
    model = get_model(config.model, pretrained=True, root=config.save_folder, local_rank=config.local_rank).to(device)
    print('Finished loading model!')
    model.eval()
    # image transform
    transform = transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for i in range(3): ret, frame = video_cap.read()
    frame_index = 0
    while True:
        ret, frame = video_cap.read()
        if not ret: break
        t0 = time.time()
        image = Image.fromarray(frame).convert("RGB")
        images = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(images)
            pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
            mask1 = get_color_pallete_c(pred, "mapillary", config.dataset)
        sys.stdout.write("\r%d/%d, time use: %.2fms" % (frame_index, 1107, (time.time()-t0)*1000))
        sys.stdout.flush()
        video_writer.write(np.array(mask1, dtype=np.uint8))
        frame_index += 1
    video_writer.release()
    print("finish")


def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print('using device %s' % device)
    # output folder
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    
    # load model
    model = get_model(config.model, pretrained=True, root=config.save_folder, local_rank=config.local_rank).to(device)
    print('Finished loading model!')
    model.eval()
    # image transform
    if len(config.input_dir) > 0:
        size = config.input_size.split(',')
        size = [int(a.strip()) for a in size]
        transform = transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)
    prefix = os.path.splitext(os.path.split(config.input_pic)[-1])[0] + "_"

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)
    if not os.path.exists(os.path.join(config.out_dir, 'raw')):
        os.mkdir(os.path.join(config.out_dir, 'raw'))
    if not os.path.exists(os.path.join(config.out_dir, 'seg_all')):
        os.mkdir(os.path.join(config.out_dir, 'seg_all'))
    if not os.path.exists(os.path.join(config.out_dir, 'seg_part')):
        os.mkdir(os.path.join(config.out_dir, 'seg_part'))

    with torch.no_grad():
        # for _ in range(REP):
        #     sstart = time.time()
        #     output = model(images)
        #     # torch.cuda.synchronize()\\
        #     # print('____time %.2fs'%(time.time()-sstart))
        #     # print('out size:', output[0].size())
        #     pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        #     start = time.time()
        #     mask1 = get_color_pallete_c(pred, "mapillary", config.dataset)
        #     mask2 = get_color_pallete_c(pred, "mapillary_full", config.dataset)
        #     prefix = os.path.splitext(os.path.split(config.input_pic)[-1])[0] + "_"
        #     outname_mask1 = prefix + config.model + '_out1.png'
        #     outname_mask2 = prefix + config.model + '_out2.png'
        #     outname_pred = prefix + config.model + '_raw'
        #     mask1.save(os.path.join(config.out_dir, outname_mask1))
        #     mask2.save(os.path.join(config.out_dir, outname_mask2))
        #     np.save(os.path.join(config.out_dir, outname_pred), pred.astype(np.int8))
        #     elapse = time.time() - start

        if config.demo_dir is not None:
            print("start dir demoing...")
            filenames = glob.glob(config.demo_dir)
            total = len(filenames)
            for ii, filename in enumerate(filenames):
                sys.stdout.write("\r%d/%d, %s" % (ii+1, total, filename))
                sys.stdout.flush()
                image = Image.open(filename).convert('RGB')
                images = transform(image).unsqueeze(0).to(device)
                output = model(images)
                pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
                mask1 = get_color_pallete_c(pred, "mapillary", config.dataset)
                # mask2 = get_color_pallete_c(pred, "mapillary_full", config.dataset)
                prefix = os.path.splitext(os.path.split(filename)[-1])[0] + "_"
                outname_mask1 = prefix + config.model + '_out1.png'
                # outname_mask2 = prefix + config.model + '_out2.png'
                outname_pred = prefix + config.model + '_raw'
                mask1.save(os.path.join(config.out_dir, 'seg_part', outname_mask1))
                # mask2.save(os.path.join(config.out_dir, 'seg_all', outname_mask2))
                np.save(os.path.join(config.out_dir, 'raw', outname_pred), pred.astype(np.int8))
        print("finish")


    # print('time used for %d repetition is %.2f seconds, %.2f seconds for each rep'%(REP, elapse, elapse/REP))

if __name__ == '__main__':
    # demo(args)
    demo_vedio(args)
