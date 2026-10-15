"""HPatches Sequences homography-estimation benchmark (Mean Homography
Accuracy). Copied from
/draft-nvme1/cadar/many_agents/src/many_agents/shared/eval/hpatches.py.

Unlike hpatches_eval.py (patch verification/matching/retrieval on the
pre-cropped patches dataset), this runs on the full-image HPatches Sequences
release (1.ppm..6.ppm + H_1_2..H_1_6 per sequence) and scores a matcher_fn's
homography estimate against the ground-truth homography.
"""
import os
import cv2
import numpy as np
import glob
from enum import Enum
import tqdm


class Split(Enum):
    ILLUMINATION = "i_"
    VIEWPOINT = "v_"


class HomographyBenchmark:
    def __init__(self, dataset_root):
        if not os.path.exists(dataset_root):
            raise RuntimeError(f'Path {dataset_root} doesnt exist.')

        self.dataset_root = dataset_root
        self.data = {'illumination': {}, 'viewpoint': {}}
        self.loaded_i = False
        self.loaded_v = False

    def read_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        return cv2.imread(path)

    def read_homography(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Homography file not found: {path}")
        return np.loadtxt(path)

    def load_split(self, split_type):
        split_folder = Split(split_type).value
        scenes = glob.glob(os.path.join(self.dataset_root, split_folder + '*'))
        for scene_path in tqdm.tqdm(scenes, desc=f'Loading {split_type}'):
            scene = os.path.basename(scene_path)
            scene_data = {'reference': None, 'targets': [], 'homographies': []}

            ref_image_path = os.path.join(scene_path, "1.ppm")
            scene_data['reference'] = self.read_image(ref_image_path)

            for i in range(2, 7):
                image_path = os.path.join(scene_path, f"{i}.ppm")
                homography_path = os.path.join(scene_path, f"H_1_{i}")
                scene_data['targets'].append(self.read_image(image_path))
                scene_data['homographies'].append(self.read_homography(homography_path))

            if 'i_' in scene:
                self.data['illumination'][f"{scene}"] = scene_data
                self.loaded_i = True
            else:
                self.data['viewpoint'][f"{scene}"] = scene_data
                self.loaded_v = True

    def run_illumination_split(self, matcher_fn):
        if not self.loaded_i:
            self.load_split(Split.ILLUMINATION)

        return self.run_MHA(matcher_fn, Split.ILLUMINATION)

    def run_viewpoint_split(self, matcher_fn):
        if not self.loaded_v:
            self.load_split(Split.VIEWPOINT)

        return self.run_MHA(matcher_fn, Split.VIEWPOINT)

    def run_MHA(self, matcher_fn, split_type, thresholds=[3, 5, 7]):
        """
            Run Mean Homography Accuracy @ thresholds
        """
        scene_t = 'viewpoint' if split_type == Split.VIEWPOINT else 'illumination'
        all_errors = []
        all_ir = []  # inlier ratio
        for scene, data in tqdm.tqdm(self.data[scene_t].items()):
            errors = []
            ref_image = data['reference']
            h, w = ref_image.shape[:2]
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            for target_image, gt_homography in zip(data['targets'], data['homographies']):
                # Step 1: Match reference frame to target frames
                mkpts0, mkpts1 = matcher_fn(ref_image, target_image)
                if len(mkpts0) > 4:
                    # Step 2: Estimate H with RANSAC/MAGSAC and threshold 3
                    estimated_H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 3., confidence=0.9999, maxIters=500)

                    # Steps 3 and 4: Warp reference image corners using GT and estimated homographies
                    warped_corners_gt = cv2.perspectiveTransform(corners, gt_homography).reshape(-1, 2)
                    try:
                        warped_corners_est = cv2.perspectiveTransform(corners, estimated_H).reshape(-1, 2)
                    except Exception:
                        print("Error in perspectiveTransform")
                        warped_corners_est = np.zeros_like(warped_corners_gt)

                    ir = np.mean(mask)
                else:
                    warped_corners_gt = cv2.perspectiveTransform(corners, gt_homography).reshape(-1, 2)
                    warped_corners_est = np.zeros_like(warped_corners_gt)
                    ir = 0.0

                # Step 6: Compute mean error for the scene
                error = np.mean(np.linalg.norm(warped_corners_gt - warped_corners_est, axis=-1))
                errors.append(error)
                all_ir.append(ir)

            # Accumulate errors
            all_errors += errors

        # Compute accuracy at specified pixel thresholds
        accuracies = {f'MHA@{t}': np.mean(np.array(all_errors) <= t) * 100 for t in thresholds}
        accuracies['MIR'] = np.mean(all_ir)

        return accuracies
