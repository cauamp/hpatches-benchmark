"""Run the HPatches Sequences Mean Homography Accuracy (MHA) benchmark
(utils/hpatches_sequences.py, copied from
/draft-nvme1/cadar/many_agents/src/many_agents/shared/eval/hpatches.py) for a
given method.

For easy_local_features methods this wires the same way extract_easy_features.py
does (easy_local_features.getMethod(elf_model, elf_conf) via
utils/easy_features.py), using the extractor's built-in `.match(img0, img1)`
(detect+describe both images, nearest-neighbor match). 'sift' is handled
separately via plain OpenCV SIFT + ratio-test matching, since it isn't part of
easy_local_features.

Usage:
    python eval_hpatches_sequences.py <hpatches_sequences_root> --elf-model alike
    python eval_hpatches_sequences.py <hpatches_sequences_root> --elf-model sift
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from utils.easy_features import get_extractor
from utils.hpatches_sequences import HomographyBenchmark


def sift_matcher_fn():
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    def match(img0, img1):
        kp0, desc0 = sift.detectAndCompute(img0, None)
        kp1, desc1 = sift.detectAndCompute(img1, None)
        if desc0 is None or desc1 is None or len(kp0) < 2 or len(kp1) < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        knn = bf.knnMatch(desc0, desc1, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        mkpts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        mkpts1 = np.float32([kp1[m.trainIdx].pt for m in good])
        return mkpts0, mkpts1

    return match


def easy_features_matcher_fn(model):
    def match(img0, img1):
        with torch.no_grad():
            out = model.match(img0, img1)
        mkpts0 = out['mkpts0'].detach().cpu().numpy()
        mkpts1 = out['mkpts1'].detach().cpu().numpy()
        return mkpts0, mkpts1

    return match


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sequences_root', help='Root folder of HPatches Sequences (1.ppm..6.ppm + H_1_2..H_1_6 per sequence dir)')
    parser.add_argument('--elf-model', required=True, help="easy_local_features extractor name, or 'sift'")
    parser.add_argument('--elf-conf', type=str, default=None, help='Path to a JSON file with extractor config overrides')
    parser.add_argument('--results-dir', type=str, default='mha_results', help='Where to write the <name>.json result file')
    parser.add_argument('--name', type=str, default=None, help='Result file stem (default: --elf-model)')
    args = parser.parse_args()
    name = args.name or args.elf_model

    if args.elf_model == 'sift':
        matcher_fn = sift_matcher_fn()
    else:
        elf_conf = {}
        if args.elf_conf:
            with open(args.elf_conf) as f:
                elf_conf = json.load(f)
        model, _ = get_extractor(args.elf_model, elf_conf)
        matcher_fn = easy_features_matcher_fn(model)

    bench = HomographyBenchmark(args.sequences_root)
    illum = bench.run_illumination_split(matcher_fn)
    view = bench.run_viewpoint_split(matcher_fn)

    print(f"[{name}] illumination: {illum}")
    print(f"[{name}] viewpoint:    {view}")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f'{name}.json'), 'w') as f:
        json.dump({'illumination': illum, 'viewpoint': view}, f, indent=2)


if __name__ == '__main__':
    main()
