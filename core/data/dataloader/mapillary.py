"""Prepare Cityscapes dataset"""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class MapillarySegmentation(SegmentationDataset):
    """Mapillary Vista Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Mapillary folder. Default is './datasets/mapillary'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples -------- >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = MapillarySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'mapillary'
    NUM_CLASS = 66
    VALID_CLASS = list(range(NUM_CLASS))

    def __init__(self, root='../datasets/mapillary', split='train', mode=None, transform=None, **kwargs):
        super(MapillarySegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please setup the dataset using ../datasets/mapillary.py"
        self.images, self.mask_paths = _get_mapillary_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = self.VALID_CLASS
        # self._key = np.array([-1, -1, -1, -1, -1, -1,
        #                       -1, -1, 0, 1, -1, -1,
        #                       2, 3, 4, -1, -1, -1,
        #                       5, -1, 6, 7, 8, 9,
        #                       10, 11, 12, 13, 14, 15,
        #                       -1, -1, 16, 17, 18])
        self._key = np.arange(-1, self.NUM_CLASS-1)
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        print("mapping: ", self._mapping)
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_mapillary_pairs(folder, split='train'):
    split_map = {'train':'training', 'val':'validation', 'test':'testing'}

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root, filename)
                    # foldername = os.path.basename(os.path.dirname(imgpath)) # get upper lever folder name
                    maskname = filename.replace('.jpg', '.png')
                    maskpath = os.path.join(mask_folder,  maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, split_map[split], 'images')
        mask_folder = os.path.join(folder, split_map[split], 'labels')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'training/images')
        train_mask_folder = os.path.join(folder, 'training/labels')
        val_img_folder = os.path.join(folder, 'validation/images')
        val_mask_folder = os.path.join(folder, 'validation/labels')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = MapillarySegmentation()
