import os
import pdb
import sys
import glob
import argparse
import time
import torch

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
    transform = transforms.Compose([
        # transforms.Resize((480,520)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)
    prefix = os.path.splitext(os.path.split(config.input_pic)[-1])[0] + "_"

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)

    with torch.no_grad():
        for _ in range(REP):
            sstart = time.time()
            output = model(images)
            # torch.cuda.synchronize()\\
            # print('____time %.2fs'%(time.time()-sstart))
            # print('out size:', output[0].size())
            pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
            start = time.time()
            mask = get_color_pallete_c(pred, config.dataset)
            pred_img = pred.astype('uint8')
            pred_img = Image.fromarray(pred_img)
            outname_mask = prefix + config.model + '_out.png'
            outname_pred = prefix + config.model + '_raw.png'
            mask.save(os.path.join(config.out_dir, outname_mask))
            pred_img.save(os.path.join(config.out_dir, outname_pred))
            elapse = time.time() - start

        if config.demo_dir is not None:
            print("start dir demoing...")
            filenames = glob.glob(config.demo_dir)
            total = len(filenames)
            for ii, filename in enumerate(filenames):
                sys.stdout.write("\r%d/%d, %s" % (ii, total, filename))
                sys.stdout.flush()
                image = Image.open(filename).convert('RGB')
                images = transform(image).unsqueeze(0).to(device)
                output = model(images)
                pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
                mask = get_color_pallete_c(pred, config.dataset)
                pred_img = pred.save
                prefix = os.path.splitext(os.path.split(filename)[-1])[0] + "_"
                outname_mask = prefix + config.model + '_out.png'
                outname_pred = prefix + config.model + '_raw.np'
                mask.save(os.path.join(config.out_dir, outname_mask))
                np.save(os.path.join(config.out_dir, outname_pred), pred)
        print("finish")


    print('time used for %d repetition is %.2f seconds, %.2f seconds for each rep'%(REP, elapse, elapse/REP))

if __name__ == '__main__':
    demo(args)
