"""Extract per-patch descriptors for the HPatches benchmark using an
easy_local_features model.

This wires easy_local_features into hpatches-benchmark the same way HLoc
wires it in HLoc/hloc/extractors/easy-features.py: the model is built with
`easy_local_features.getMethod(elf_model, elf_conf)` (see utils/easy_features.py)
and run on the best available device. Unlike hloc's full-image
detect-and-compute, HPatches patches are already cropped to a fixed size
around a single keypoint, so we call the extractor's `compute(image,
keypoints)` with one keypoint at the patch center -- the same approach
extract_opencv_sift.py uses with OpenCV SIFT.

Usage:
    python extract_easy_features.py <hpatches_db_root> --elf-model alike
    python extract_easy_features.py <hpatches_db_root> --elf-model aliked \
        --elf-conf aliked_conf.json --out-size 65
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from easy_local_features.feature.basemodel import MethodType
from utils.easy_features import get_extractor

# all types of patches
tps = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5',
       't1', 't2', 't3', 't4', 't5']


class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps

    def __init__(self, base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t + '.png')
            im = cv2.imread(im_path, 0)
            self.N = im.shape[0] / 65
            setattr(self, t, np.split(im, self.N))


def to_rgb_patch(patch, out_size):
    """Resize a grayscale HPatches patch and replicate it to 3 channels,
    matching how hloc always reads images in color for the easy-features
    extractors (see grayscale=False in HLoc/hloc/extract_features.py)."""
    if out_size != patch.shape[0]:
        patch = cv2.resize(patch, (out_size, out_size))
    return np.stack([patch] * 3, axis=-1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'hpatches_root',
        help='Root folder of HPatches sequences (e.g. ../data/hpatches-release)')
    parser.add_argument(
        '--elf-model', required=True,
        help='easy_local_features extractor name (see easy_local_features.available_extractors)')
    parser.add_argument(
        '--elf-conf', type=str, default=None,
        help='Path to a JSON file with extractor config overrides')
    parser.add_argument(
        '--out-size', type=int, default=65,
        help='Patch size fed to the extractor (default: native 65px)')
    parser.add_argument(
        '--output-name', type=str, default=None,
        help='Descriptor output folder name (default: easy-<elf-model>-<out-size>)')
    args = parser.parse_args()

    elf_conf = {}
    if args.elf_conf:
        with open(args.elf_conf) as f:
            elf_conf = json.load(f)

    model, device = get_extractor(args.elf_model, elf_conf)

    if model.method_type == MethodType.END2END_MATCHER:
        sys.exit(
            f"'{args.elf_model}' is an end-to-end matcher (no per-keypoint "
            "descriptor); it can't be used for HPatches patch descriptor "
            "extraction. Pick a detect_describe or descriptor_only extractor.")

    descr_name = args.output_name or f'easy-{args.elf_model}-{args.out_size}'

    seqs = sorted(
        os.path.abspath(p) for p in glob.glob(os.path.join(args.hpatches_root, '*'))
        if os.path.isdir(p))

    c = args.out_size / 2.0
    center_kp = np.array([[c, c]], dtype=np.float32)

    for seq_path in tqdm(seqs, desc='sequences'):
        seq = hpatches_sequence(seq_path)
        out_dir = os.path.join(descr_name, seq.name)
        os.makedirs(out_dir, exist_ok=True)
        for tp in tps:
            out_file = os.path.join(out_dir, tp + '.csv')
            if os.path.isfile(out_file):
                continue
            descs = []
            with torch.no_grad():
                for patch in getattr(seq, tp):
                    rgb_patch = to_rgb_patch(patch, args.out_size)
                    out = model.compute(rgb_patch, center_kp)
                    if isinstance(out, tuple):
                        out = out[1]
                    desc = out.detach().float().cpu().numpy().reshape(-1)
                    descs.append(desc)
            np.savetxt(out_file, np.stack(descs, axis=0), delimiter=',', fmt='%10.5f')


if __name__ == '__main__':
    main()
