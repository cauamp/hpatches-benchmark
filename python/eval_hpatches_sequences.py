"""Run the HPatches Sequences Mean Homography Accuracy (MHA) benchmark
(utils/hpatches_sequences.py, copied from
/draft-nvme1/cadar/many_agents/src/many_agents/shared/eval/hpatches.py) for a
given method.

For easy_local_features methods this wires the same way extract_easy_features.py
does (easy_local_features.getMethod(elf_model, elf_conf) via
utils/easy_features.py), using the extractor's built-in `.match(img0, img1)`
(detect+describe both images, nearest-neighbor match). 'sift' is handled
separately via plain OpenCV SIFT + ratio-test matching, since it isn't part of
easy_local_features. 'zippypoint' is TensorFlow-based (hloc's own extractor,
not easy_local_features) and MUST be run with the dedicated
.venv-zippypoint/bin/python interpreter -- the main venv's torch/kornia/larq
combination is broken for it (see HLoc/hloc/extractors/zippypoint.py). Model
imports for each backend are lazy so this file stays importable under either
interpreter.

Usage:
    python eval_hpatches_sequences.py <hpatches_sequences_root> --elf-model alike
    python eval_hpatches_sequences.py <hpatches_sequences_root> --elf-model sift
    .venv-zippypoint/bin/python eval_hpatches_sequences.py <root> --elf-model zippypoint
"""
import argparse
import json
import os

import cv2
import numpy as np

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
    import torch

    def match(img0, img1):
        with torch.no_grad():
            out = model.match(img0, img1)
        mkpts0 = out['mkpts0'].detach().cpu().numpy()
        mkpts1 = out['mkpts1'].detach().cpu().numpy()
        return mkpts0, mkpts1

    return match


def zippypoint_matcher_fn():
    """hloc's own ZippyPoint extractor (HLoc/hloc/extractors/zippypoint.py),
    not part of easy_local_features. Descriptors are 256 binary bits; packed
    to bytes and matched with Hamming distance + mutual (crossCheck) nearest
    neighbor, the same convention hloc uses ('Hamming-mutual' matcher_conf in
    efficiency/resume_after_reboot_cmu.py).
    """
    import os
    import sys
    from pathlib import Path
    # ZippyPoint's own code does `from utils.utils import ...`. Its utils/ has
    # no __init__.py (namespace-package portion only), while this repo's own
    # utils/ is a regular package -- a regular package always wins over a
    # namespace portion regardless of sys.path order, and it's already
    # cached in sys.modules by the time we get here. So: pop the cached
    # `utils*` modules AND temporarily drop this script's own directory from
    # sys.path (the thing that makes our `utils` a regular-package hit) for
    # the duration of the import. Same spirit as easy_local_features' fix for
    # an analogous 'modules' collision loading XFeat (see
    # baseline_xfeat.py's _hub_load_xfeat).
    stale = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'utils' or k.startswith('utils.')}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != script_dir]
        sys.path.insert(0, str(Path(
            '/tank/teams/image_matching_team/cauamp/proc9draft/xfeatpp/xfeat-tpami25-src/HLoc/third_party/ZippyPoint')))
        import zippypoint
    finally:
        sys.path[:] = original_path
        for k, v in stale.items():
            sys.modules.setdefault(k, v)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def to_packed(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = zippypoint.detectAndCompute(rgb)
        kpts = data['keypoints'][0].numpy()
        bits = data['descriptors'][0].numpy().astype(np.uint8)
        packed = np.packbits(bits.reshape(bits.shape[0], -1, 8), axis=-1).reshape(bits.shape[0], -1)
        return kpts, packed

    def match(img0, img1):
        kpts0, desc0 = to_packed(img0)
        kpts1, desc1 = to_packed(img1)
        if len(kpts0) == 0 or len(kpts1) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        matches = bf.match(desc0, desc1)
        mkpts0 = np.float32([kpts0[m.queryIdx] for m in matches])
        mkpts1 = np.float32([kpts1[m.trainIdx] for m in matches])
        return mkpts0, mkpts1

    return match


def hfnet_matcher_fn(max_keypoints=4096,
                      client_dir='/home/phmgc/projects/TPAMI/xfeat-tpami25-src/matching_benchmarks/hfnet'):
    """hloc's HFNet extractor talks to a TF1 server over a localhost TCP
    socket (see hloc/extractors/hfnet.py, matching_benchmarks/hfnet/). The
    server (TF1, singularity container, weights baked in) must already be
    running on this same node before this matcher is used -- see
    run_hfnet_server_proc9.slurm, which starts it before calling this script.
    Mutual nearest-neighbor matching on local_descriptors, done directly in
    numpy (same result as hfnet_wrapper.py's kornia DescriptorMatcher('mnn')).
    """
    import sys
    sys.path.insert(0, client_dir)
    from hfnet_client import HFNetClient

    client = HFNetClient()

    def match(img0, img1):
        # hfnet_server.py's inference() does image[..., ::-1] internally to
        # get RGB for the network, so pass raw BGR straight through (same as
        # hfnet_wrapper.py's HFNetWrapper.match, which does no conversion).
        out0 = client.inference(img0, num_keypoints=max_keypoints)
        out1 = client.inference(img1, num_keypoints=max_keypoints)
        kpts0, desc0 = out0['keypoints'], out0['local_descriptors']
        kpts1, desc1 = out1['keypoints'], out1['local_descriptors']
        if len(kpts0) == 0 or len(kpts1) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        d0 = desc0 / (np.linalg.norm(desc0, axis=1, keepdims=True) + 1e-8)
        d1 = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
        sim = d0 @ d1.T
        nn01 = sim.argmax(axis=1)
        nn10 = sim.argmax(axis=0)
        mutual = nn10[nn01] == np.arange(len(d0))
        idx0 = np.where(mutual)[0]
        idx1 = nn01[idx0]
        return kpts0[idx0].astype(np.float32), kpts1[idx1].astype(np.float32)

    return match


def liftfeat_matcher_fn(max_keypoints=4096,
                        liftfeat_root='/tank/teams/image_matching_team/cauamp/proc9draft/LiftFeat'):
    """LiftFeat (ICRA 2025), not part of easy_local_features. Uses its own
    wrapper (models/liftfeat_wrapper.py) with the official evals' defaults
    (weights/LiftFeat.pth, detect_threshold=0.1), capped to max_keypoints and its built-in mutual
    nearest-neighbor match_liftfeat, which takes BGR uint8 like cv2.imread.

    LiftFeat ships top-level `utils` and `models` packages, and its `utils`
    collides with this repo's: swap our cached `utils*` modules out while
    importing it, then put them back (LiftFeat's modules keep their own
    references to what they imported).
    """
    import sys

    def owned(k):
        return k in ('utils', 'models') or k.startswith(('utils.', 'models.'))

    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if owned(k)}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = list(sys.path)
    try:
        sys.path[:] = [liftfeat_root] + [p for p in sys.path if os.path.abspath(p or '.') != script_dir]
        from models.liftfeat_wrapper import LiftFeat
    finally:
        sys.path[:] = original_path
        for k in [k for k in sys.modules if owned(k)]:
            sys.modules.pop(k)
        sys.modules.update(ours)

    model = LiftFeat(top_k=max_keypoints)
    # The wrapper stores top_k but its NMS never applies it (keeps every
    # keypoint above detect_threshold), so cap to the top_k highest-scoring
    # keypoints here, as the other methods' confs do. match_liftfeat calls
    # self.extract, so wrapping it on the instance is enough.
    extract = model.extract

    def extract_top_k(image):
        out = extract(image)
        idx = out['scores'].argsort(descending=True)[:model.top_k]
        return {k: v[idx] for k, v in out.items()}

    model.extract = extract_top_k

    def match(img0, img1):
        mkpts0, mkpts1 = model.match_liftfeat(img0, img1)
        return mkpts0.astype(np.float32), mkpts1.astype(np.float32)

    return match


def upal_matcher_fn(weights_path, device):
    import torch
    from upal import load_model
    from upal.postprocess import mutual_nearest_neighbors

    model = load_model(weights_path, device=device)

    def to_tensor(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)

    def match(img0, img1):
        with torch.inference_mode():
            pred0 = model(to_tensor(img0))
            pred1 = model(to_tensor(img1))
        idx = mutual_nearest_neighbors(pred0['descriptors'][0], pred1['descriptors'][0]).detach().cpu().numpy()
        if len(idx) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        kpts0 = pred0['keypoints'][0].detach().cpu().numpy()
        kpts1 = pred1['keypoints'][0].detach().cpu().numpy()
        return kpts0[idx[:, 0]], kpts1[idx[:, 1]]

    return match


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sequences_root', help='Root folder of HPatches Sequences (1.ppm..6.ppm + H_1_2..H_1_6 per sequence dir)')
    parser.add_argument('--elf-model', required=True, help="easy_local_features extractor name, 'sift', 'upal', 'zippypoint', 'hfnet', or 'liftfeat'")
    parser.add_argument('--elf-conf', type=str, default=None, help='Path to a JSON file with extractor config overrides')
    parser.add_argument('--upal-weights', type=str, default='/tank/teams/image_matching_team/cauamp/proc9draft/upal/weights/upal.tar',
                         help='Path to the local UPAL checkpoint (only used with --elf-model upal)')
    parser.add_argument('--results-dir', type=str, default='mha_results', help='Where to write the <name>.json result file')
    parser.add_argument('--name', type=str, default=None, help='Result file stem (default: --elf-model)')
    args = parser.parse_args()
    name = args.name or args.elf_model

    if args.elf_model == 'sift':
        matcher_fn = sift_matcher_fn()
    elif args.elf_model == 'zippypoint':
        matcher_fn = zippypoint_matcher_fn()
    elif args.elf_model == 'hfnet':
        matcher_fn = hfnet_matcher_fn()
    elif args.elf_model == 'liftfeat':
        matcher_fn = liftfeat_matcher_fn()
    elif args.elf_model == 'upal':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        matcher_fn = upal_matcher_fn(args.upal_weights, device)
    else:
        from utils.easy_features import get_extractor
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
