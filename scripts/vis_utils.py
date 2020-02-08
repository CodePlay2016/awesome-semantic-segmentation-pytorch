import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import time
import imageio
import cv2


#############################################################
# names of labels
#############################################################
LABEL_NAMES_MAPILLARY_PARTIAL = np.asarray([
    "drivable area", "curb", "human", "obstacles", "vehicle", "terrain", "other"
])
LABEL_NAMES_CITYSCAPES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle'
])


#############################################################
# the label maps from prediction to needed
#############################################################
LABEL_MAP_CITYSCAPES = [-1, -1, -1, -1, -1, -1,
                        -1, -1, 0, 1, -1, -1,
                        2, 3, 4, -1, -1, -1,
                        5, -1, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15,
                        -1, -1, 16, 17, 18]
LABEL_MAP_MAPILLARY  = [-1, -1, 1, 2, 2, 2, -1, 0, 0, 1,
                        0, 6, 0, 0, 0, -1, -1, -1, -1, 5,
                        7, 7, 7, 0, 0, -1, -1, -1, -1, -1,
                        -1,-1, -1, -1, 8, -1, 0, -1, 8, -1,
                        -1, 0, 8, 0, -1, 8, -1, 8, -1, -1,
                        -1, 8, 3, 4, 4, 4, 4, 3, 4, 4,
                        4, 4, 4, -1, -1, -1]


#############################################################
# color maps
#############################################################
COLORMAP_MAPILLARY_PARTIAL = np.array([40, 190, 120,     220, 220, 0,
                              0, 255, 255, 0, 0, 255,
                              255, 0, 0,   255, 0, 255,
                              0, 255, 0,   0, 0, 0], dtype=int).reshape((-1, 3))
COLORMAP_CITYSCAPES = np.array([128, 64, 128,
                        244, 35, 232,
                        70, 70, 70,
                        102, 102, 156,
                        190, 153, 153,
                        153, 153, 153,
                        250, 170, 30,
                        220, 220, 0,
                        107, 142, 35,
                        152, 251, 152,
                        0, 130, 180,
                        220, 20, 60,
                        255, 0, 0,
                        0, 0, 142,
                        0, 0, 70,
                        0, 60, 100,
                        0, 80, 100,
                        0, 0, 230,
                        119, 11, 32],dtype=int).reshape((-1,3))
def create_pascal_label_colormap(num_classes):
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((num_classes, 3), dtype=int)
    ind = np.arange(num_classes, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def get_label_name(name):
    if name == "mapillary_partial":
        return LABEL_NAMES_MAPILLARY_PARTIAL
    elif name == "cityscapes":
        return LABEL_NAMES_CITYSCAPES


def get_colormap(colormap_name):
    if colormap_name == "pascal":
        return create_pascal_label_colormap(256)
    elif colormap_name == "mapillary_partial":
        return COLORMAP_MAPILLARY_PARTIAL
    elif colormap_name == "cityscapes":
        return COLORMAP_CITYSCAPES


def get_maskmap(data_name):
    if data_name == "mapillary_partial":
        return LABEL_MAP_MAPILLARY
    elif data_name == "cityscapes":
        return LABEL_MAP_CITYSCAPES


def label_to_color_image(label, data_name):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = get_colormap(data_name)

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


#############################################################
# do some drawing
#############################################################
def vis_segmentation(image, seg_map, seg_image, colormap, label_names):
    """Visualizes input image, segmentation map and overlay view."""
    fig = plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    ax1 = fig.add_subplot(grid_spec[0])
    ax1.imshow(image)
    ax1.axis('off')
    plt.title('input image')

    ax2 = fig.add_subplot(grid_spec[1])
    ax2.imshow(seg_image)
    ax2.axis('off')
    plt.title('segmentation map')

    ax3 = fig.add_subplot(grid_spec[2])
    ax3.imshow(image)
    ax3.imshow(seg_image, alpha=0.7)
    ax3.axis('off')
    plt.title('segmentation overlay')
    

    unique_labels = np.unique(seg_map)
    ax4 = fig.add_subplot(grid_spec[3])
    ax4.imshow(colormap[unique_labels].astype(np.uint8), interpolation='nearest')
    ax4.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), label_names[unique_labels])
    plt.xticks([], [])
    ax4.tick_params(width=0.0)
    bbox = ax4.get_tightbbox(fig.canvas.get_renderer())
    # fig.savefig("subplot_{}.png".format(time.strftime("%m%d_%H%M%S", time.localtime(time.time()))),
    #             bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))
    # fig.savefig("./F/subplot_%.4f.png"%(time.time()),
    #             bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()))
    plt.grid('off')
    # plt.show()
    return fig, ax4


def visualization(image, pred, data_name, map_pred=False, add_on=False):
    """Inferences DeepLab model and visualizes result."""
    if map_pred:
        pred = map_mask(pred, get_maskmap(data_name))
    seg_image = label_to_color_image(pred, data_name).astype(np.uint8)
    pred = pred.astype(np.int32)

    label_names = get_label_name(data_name)
    FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
    colormap = label_to_color_image(FULL_LABEL_MAP, data_name)
    if add_on:
        seg_image = cv2.resize(seg_image, image.shape[1::-1])
        seg_image = cv2.addWeighted(image, 0.3, seg_image, 0.7, 0)
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    return seg_image
    

def map_mask(seg_mask, mask_map):
    unique = np.unique(seg_mask)
    mapped_seg_mask = np.zeros_like(seg_mask)
    for i in unique:
        mapped_seg_mask = np.where(np.equal(seg_mask, i), np.ones_like(seg_mask)*mask_map[i], mapped_seg_mask)
    return mapped_seg_mask


def make_gif(image_list, out_path):
    imageio.mimsave(out_path, image_list, format='GIF', duration=5)

def vis_test():
    img_path = "/Users/hufangquan/self/AIWAYS/projects/Low-obstacle_detection/fisheye_data/bbb_correct/snapshot_14_5120x720_B.png"
    pred_path = "/Users/hufangquan/self/AIWAYS/projects/Low-obstacle_detection/fisheye_data/eval_fish_eye_correct/snapshot_14_5120x720_B_deeplabv3_resnet101_mapillary_raw.npy"
    data_name = "mapillary_partial"
    resize_shape = (1280, 720)
    original_im = Image.open(img_path)
    resized_im = original_im.convert('RGB').resize(resize_shape, Image.ANTIALIAS)
    seg_map = np.load(pred_path)
    seg_map = map_mask(seg_map, get_maskmap(data_name))

    seg_image = label_to_color_image(seg_map, data_name).astype(np.uint8)

    label_names = get_label_name(data_name)
    FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
    colormap = label_to_color_image(FULL_LABEL_MAP, data_name)
    seg_map = seg_map.astype(np.int32)
    label_names = get_label_name(data_name)
    vis_segmentation(resized_im, seg_map, seg_image, colormap, label_names)

def draw_colormap():
    pallete_np = get_colormap('cityscapes')
    names = get_label_name('cityscapes')
    plt.figure()
    plt.imshow(pallete_np.reshape((-1,1,3)), interpolation='nearest')
    plt.yticks(range(len(names)), names, fontsize=10)
    plt.xticks([], [])
    plt.tick_params(width=0.0)
    plt.show()


def add_semi_transparent(img, mask, alpha=0.7):
    added = cv2.addWeighted(img, 1-alpha, mask, alpha, 0)
    return added

if __name__ == "__main__":
    print('hhh')
    img_dir = '/Users/hufangquan/self/AIWAYS/projects/Low-obstacle_detection/fisheye_data/correct_cylinder'
    mask_dir = '/Users/hufangquan/self/AIWAYS/projects/Low-obstacle_detection/fisheye_data/correct_cylinder/out_res152_mapi_321/seg_all'
    out_dir = '/Users/hufangquan/self/AIWAYS/projects/Low-obstacle_detection/fisheye_data/correct_cylinder/out_res152_mapi_321/masked_all'
    import os, glob
    img_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    mask_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    for ii in range(len(img_list)):
        img_path = img_list[ii]
        mask_path = mask_list[ii]
        img = np.array(Image.open(img_path))
        width, height, _ = img.shape
        mask = Image.open(mask_path)
        mask = mask.resize((height, width), Image.ANTIALIAS)
        mask = np.array(mask)
        added = add_semi_transparent(img, mask, 0.7)
        added = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out_dir, os.path.split(img_path)[-1]), added)


