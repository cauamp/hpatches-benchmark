"""Run the two-view relative pose benchmark (MegaDepth-1500 / ScanNet-1500;
utils/pose_eval.py, copied from image-matching_eval) for a given method.

Same method wiring as eval_hpatches_sequences.py (whose matcher functions are
reused here), same protocol as image-matching_eval's run_pose_variants.py:
full-resolution images, poselib RANSAC swept over thresholds 0.5..6.0 px, and
AUC@5/10/20 deg reported as the best over that sweep. easy_local_features
methods get the RGB tensor directly; the other backends take BGR uint8 like
cv2.imread. Run zippypoint with .venv-zippypoint/bin/python (see
eval_hpatches_sequences.py).

Usage:
    python eval_pose.py --dataset megadepth --elf-model alike --elf-conf confs/alike_fullres.json
"""
import argparse
import json
import os

import cv2
import numpy as np

import eval_hpatches_sequences as seq
from utils.pose_eval import PoseEval

DATASETS = {
    'megadepth': '/srv/storage/datasets/cauamagalhaes/datasets/megadepth1500',
    'scannet': '/srv/storage/datasets/cauamagalhaes/datasets/scannet_test_1500',
}
POSE_TH = [5, 10, 20]


def to_bgr_uint8(image):
    rgb = (image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_matcher(args):
    if args.elf_model in ('sift', 'zippypoint', 'hfnet', 'upal'):
        if args.elf_model == 'sift':
            bgr_match = seq.sift_matcher_fn()
        elif args.elf_model == 'zippypoint':
            bgr_match = seq.zippypoint_matcher_fn()
        elif args.elf_model == 'hfnet':
            bgr_match = seq.hfnet_matcher_fn()
        else:
            import torch
            bgr_match = seq.upal_matcher_fn(args.upal_weights, 'cuda' if torch.cuda.is_available() else 'cpu')

        def match(image0, image1):
            mkpts0, mkpts1 = bgr_match(to_bgr_uint8(image0), to_bgr_uint8(image1))
            return {'mkpts0': mkpts0, 'mkpts1': mkpts1}
        return match

    import torch
    from utils.easy_features import get_extractor
    elf_conf = {}
    if args.elf_conf:
        with open(args.elf_conf) as f:
            elf_conf = json.load(f)
    model, _ = get_extractor(args.elf_model, elf_conf)

    def match(image0, image1):
        with torch.no_grad():
            out = model.match(image0, image1)
        return {'mkpts0': out['mkpts0'], 'mkpts1': out['mkpts1']}
    return match


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=list(DATASETS), required=True)
    parser.add_argument('--elf-model', required=True, help="easy_local_features extractor name, 'sift', 'upal', 'zippypoint', or 'hfnet'")
    parser.add_argument('--elf-conf', type=str, default=None, help='Path to a JSON file with extractor config overrides')
    parser.add_argument('--upal-weights', type=str, default='/tank/teams/image_matching_team/cauamp/proc9draft/upal/weights/upal.tar')
    parser.add_argument('--results-dir', type=str, default='pose_results', help='Where to write <dataset>/<name>.json')
    parser.add_argument('--name', type=str, default=None, help='Result file stem (default: --elf-model)')
    parser.add_argument('--n-workers', type=int, default=16)
    args = parser.parse_args()
    name = args.name or args.elf_model

    root = DATASETS[args.dataset]
    ev = PoseEval({
        'data_path': root, 'pairs_path': os.path.join(root, 'pairs_calibrated.txt'),
        'output': os.path.join(args.results_dir, 'work', args.dataset),
        'pose_estimator': 'poselib', 'pose_thresholds': POSE_TH,
        'n_workers': args.n_workers, 'plot_matches': False,
    })
    summary = ev.run_benchmark(build_matcher(args), name=name)[0]
    abt = summary['aucs_by_thresh']
    best = {t: max(abt[rt][t] for rt in abt) for t in POSE_TH}
    print(f"[{name}] {args.dataset} ({len(ev.pairs)} pairs) best AUC@5/10/20: "
          + ' / '.join(f'{best[t]:.2f}' for t in POSE_TH))

    out_dir = os.path.join(args.results_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump({'best_auc': best, 'aucs_by_thresh': abt, 'accs_by_thresh': summary['accs_by_thresh']}, f, indent=2)


if __name__ == '__main__':
    main()
