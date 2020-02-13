import torch
import torch.onnx
import torch.nn as nn

import os, sys
cur_path = os.path.abspath(os.path.dirname("."))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from core.models.model_zoo import get_segmentation_model
from core.models import get_model

device = torch.device("cpu")
# weights_path = "/home/hufq/LOD/awesome-semantic-segmentation-pytorch/model/mapillary_selected/deeplabv3_resnet50_mapillary.pth"
weights_path = "../model/mapillary_test/deeplabv3_mobilenetv2_mapillary.pth"
# weights_path = "../model/mapillary_test/deeplabv3_resnet18_mapillary.pth"
weights_dir = "/".join(weights_path.split("/")[:-1])
weights_name  = weights_path.split("/")[-1].split('.')[0]
export_weights_path = os.path.join(weights_dir, weights_name+".onnx")

# create network
BatchNorm2d = nn.SyncBatchNorm
model = get_model(weights_name, pretrained=True, pre_conv=True, do_aspp=False, export=True,
                  root=weights_dir, local_rank=0).to(device)
model.eval()

# Create the right input shape (e.g. for an image)
sample_batch_size = 1
channel = 3
height = 320
width = 640
dummy_input = torch.randn(sample_batch_size, channel, height, width)

torch.onnx.export(model, dummy_input, export_weights_path, input_names=['input_image'], output_names=['seg_result'],
                  opset_version=11)
# torch.onnx.export(model, dummy_input, export_weights_path, input_names=['input_image'], output_names=['seg_result'])
