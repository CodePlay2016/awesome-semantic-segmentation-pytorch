"""Prepare Cityscapes dataset"""
import os
import sys
import argparse
import zipfile

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)

from core.utils import download, makedirs, check_sha1

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/mapillary')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize mapillary dataset.',
        epilog='Example: python prepare_mapillary.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args


def download_mapillary(path, overwrite=False):
    # no valid link to download
    print("!!!!!!!!!!!!!!!!no valid link to download mapillary dataset. use local dataset instead")
    return

if __name__ == '__main__':
    args = parse_args()
    makedirs(os.path.expanduser('~/.torch/datasets'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)
    else:
        download_mapillary(_TARGET_DIR, overwrite=False)
