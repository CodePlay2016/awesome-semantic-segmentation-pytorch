import os
import sys
import argparse
import time
import torch

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete, get_color_pallete_c
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc','pascal_aug','ade20k','citys'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str, default='../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg',
                    help='path to the input picture')
parser.add_argument('--out-dir', default='./eval', type=str,
                    help='path to save the predict result')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--repetition', type=int, default=1)
args = parser.parse_args()

REP = args.repetition

def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(config.input_pic).convert('RGB')
    images = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        for _ in range(REP):
                output = model(images)
            # pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
            # mask = get_color_pallete_c(pred, config.dataset)
            # outname = os.path.splitext(os.path.split(config.input_pic)[-1])[0] + config.model + '.png'
            # mask.save(os.path.join(config.out_dir, outname))
        elapse = time.time() - start
    print('time used for %d repetition is %.2f seconds, %.2f seconds for each rep'%(REP, elapse, elapse/REP))

if __name__ == '__main__':
    demo(args)
