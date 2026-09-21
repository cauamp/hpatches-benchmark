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


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _descriptor_of(out):
    if isinstance(out, tuple):
        out = out[1]
    return out.detach().float().cpu().numpy()


def compute_via_detect(model, patch, c, desc_dim_hint):
    """Fallback for models whose compute() is unimplemented (e.g. SFD2 in
    this easy_local_features version): run full detectAndCompute() on the
    patch -- the same call hloc itself always uses -- and take the detected
    keypoint nearest to the patch center as the patch's descriptor.
    """
    out = model.detectAndCompute(patch, return_dict=True)
    kpts = out['keypoints'].detach().float().cpu().numpy().reshape(-1, 2)
    descs = out['descriptors'].detach().float().cpu().numpy().reshape(kpts.shape[0], -1)
    if kpts.shape[0] == 0:
        dim = descs.shape[-1] if descs.size else desc_dim_hint
        return np.zeros(dim, dtype=np.float32)
    dists = np.sum((kpts - np.array([c, c])) ** 2, axis=1)
    return descs[np.argmin(dists)]


def compute_batch(model, rgb_batch, c, device):
    """Compute descriptors for a batch of same-size patches at their center.

    Keypoints are passed as a torch.Tensor (not numpy) because some baselines
    (e.g. SuperPoint) call tensor-only ops on them without converting first.

    Some easy_local_features baselines' compute() doesn't actually support
    batch size > 1 correctly (e.g. SuperPoint's internal per-image keypoint
    detection returns a variable count per patch and fails to stack; DISK
    hardcodes a batch size of 1 internally). Rather than special-case each
    model, try the batched call and fall back to one-patch-at-a-time on
    failure.
    """
    b = len(rgb_batch)
    kp_batch = torch.full((b, 1, 2), c, dtype=torch.float32, device=device)
    try:
        desc = _descriptor_of(model.compute(np.stack(rgb_batch, axis=0), kp_batch))
        return desc.reshape(b, -1)
    except Exception:
        descs = []
        for patch, kp in zip(rgb_batch, kp_batch):
            desc = _descriptor_of(model.compute(patch, kp))
            descs.append(desc.reshape(-1))
        return np.stack(descs, axis=0)


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
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Patches per forward pass (compute() accepts batched images/keypoints '
             'per the BaseExtractor contract; batching is needed to make extraction '
             'over the ~2.5M HPatches patches tractable). Default: 128.')
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

    # Probe whether compute() is actually implemented (some baselines, e.g.
    # SFD2 in this easy_local_features version, declare it but raise
    # NotImplementedError). If not, fall back to full detectAndCompute() per
    # patch and take the keypoint nearest the patch center.
    dummy = to_rgb_patch(np.zeros((args.out_size, args.out_size), dtype=np.uint8), args.out_size)
    with torch.no_grad():
        try:
            model.compute(dummy, torch.tensor([[[c, c]]], dtype=torch.float32, device=device))
            use_detect_fallback = False
        except NotImplementedError:
            use_detect_fallback = True
            print(f"'{args.elf_model}'.compute() is not implemented; falling back to "
                  "detectAndCompute() + nearest-keypoint-to-center per patch.")

    for seq_path in tqdm(seqs, desc='sequences'):
        seq = hpatches_sequence(seq_path)
        out_dir = os.path.join(descr_name, seq.name)
        os.makedirs(out_dir, exist_ok=True)
        for tp in tps:
            out_file = os.path.join(out_dir, tp + '.csv')
            if os.path.isfile(out_file):
                continue
            patches = getattr(seq, tp)
            descs = []
            with torch.no_grad():
                if use_detect_fallback:
                    for patch in patches:
                        rgb_patch = to_rgb_patch(patch, args.out_size)
                        descs.append(compute_via_detect(model, rgb_patch, c, None))
                else:
                    for batch in chunks(patches, args.batch_size):
                        rgb_batch = [to_rgb_patch(p, args.out_size) for p in batch]
                        descs.append(compute_batch(model, rgb_batch, c, device))
            np.savetxt(out_file, np.stack(descs, axis=0) if use_detect_fallback
                       else np.concatenate(descs, axis=0), delimiter=',', fmt='%10.5f')


if __name__ == '__main__':
    main()
