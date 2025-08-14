#!/usr/bin/env python3
"""
Feature extraction script for clinical videos/audios.
Upgrades body & face keypoints to higher-accuracy models.

- Body keypoints (COCO-17): ViTPose-H (huge) 256x192
- Face keypoints (WFLW-98): HRNetv2-W18 + AdaptiveWingLoss 256x256
- Detector: RTMDet-L (COCO)

Saves features as PyTorch tensors (.pt):
  - pose:  [num_frames, max_persons, K_body(=17), 3]  (x,y,conf)
  - face:  [num_frames, max_faces,  K_face(=98),  3]  (x,y,conf)
  - opensmile: [1, D]  (functionals)

This script uses local mmpose configs to avoid relative referencing issues.
"""

import argparse
import json
import os
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import subprocess
import sys

warnings.filterwarnings('ignore')

# OpenSmile
import opensmile

# MMPose and MMDetection
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.utils import adapt_mmdet_pipeline


# -------------------------------
# Setup: Clone repositories
# -------------------------------

def setup_repos(base_dir=None):
    """Clone mmpose and mmdetection repositories if not present."""
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    mmpose_dir = base_dir / 'mmpose'
    mmdet_dir = base_dir / 'mmdetection'

    # Clone mmpose if not exists
    if not mmpose_dir.exists():
        print("Cloning mmpose repository...")
        subprocess.run([
            'git', 'clone',
            'https://github.com/open-mmlab/mmpose.git',
            str(mmpose_dir)
        ], check=True)
        print(f"MMPose cloned to {mmpose_dir}")
    else:
        print(f"MMPose already exists at {mmpose_dir}")

    # Clone mmdetection if not exists
    if not mmdet_dir.exists():
        print("Cloning mmdetection repository...")
        subprocess.run([
            'git', 'clone',
            'https://github.com/open-mmlab/mmdetection.git',
            str(mmdet_dir)
        ], check=True)
        print(f"MMDetection cloned to {mmdet_dir}")
    else:
        print(f"MMDetection already exists at {mmdet_dir}")

    return mmpose_dir, mmdet_dir


# -------------------------------
# Audio: OpenSmile features
# -------------------------------

def extract_opensmile_features(audio_path, output_path):
    """Extract OpenSmile ComParE_2016 functionals and save as a tensor."""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_file(str(audio_path))
        feature_tensor = torch.tensor(features.values, dtype=torch.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_tensor, output_path)
        return True
    except Exception as e:
        print(f"Error extracting OpenSmile features from {audio_path}: {e}")
        return False


# -------------------------------
# Video: Pose & Face keypoints
# -------------------------------

def _get_person_bboxes(det_result, score_thr=0.3):
    """Return Nx4 numpy array of person boxes from MMDet result across versions."""
    bboxes = np.zeros((0, 4), dtype=np.float32)

    # MMDet >=3.0 returns DetDataSample with .pred_instances
    if hasattr(det_result, 'pred_instances'):
        pred = det_result.pred_instances
        if hasattr(pred, 'bboxes') and hasattr(pred, 'labels'):
            keep = (pred.labels == 0)
            boxes = pred.bboxes[keep]
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            bboxes = boxes[:, :4] if boxes.size else bboxes
        return bboxes

    # Older MMDet returns tuple/list of arrays; class 0 is 'person'
    try:
        arr = det_result[0][0]
        if arr is not None and len(arr) > 0:
            arr = np.asarray(arr)
            if arr.shape[1] >= 4:
                if arr.shape[1] == 4:
                    bboxes = arr[:, :4]
                else:
                    # [x1,y1,x2,y2,score] -> filter by score
                    keep = arr[:, 4] >= score_thr
                    bboxes = arr[keep, :4]
    except Exception:
        pass

    return bboxes.astype(np.float32)


def _keypoint_count_from_model(model, fallback=17):
    """Best-effort to read keypoint count from an MMPose model or its config."""
    try:
        # Try dataset_meta first (MMPose 1.x)
        meta = getattr(model, 'dataset_meta', None)
        if isinstance(meta, dict):
            if 'num_keypoints' in meta:
                return int(meta['num_keypoints'])
            if 'keypoint_colors' in meta:
                # Sometimes len of colors == K
                return int(len(meta['keypoint_colors']))

        # Try config
        cfg = getattr(model, 'cfg', None)
        if cfg is not None and 'model' in cfg:
            head = cfg['model'].get('head', {})
            for k in ('out_channels', 'num_joints', 'num_keypoints'):
                if k in head:
                    return int(head[k])
    except Exception:
        pass
    return int(fallback)


def extract_pose_features(video_path, output_path, pose_model, detector, body_kpts=17):
    """Extract body keypoints from video using MMPose top-down with person detector."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0

        all_poses = []

        with tqdm(total=total_frames, desc="  Extracting pose frames", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                det_result = inference_detector(detector, frame)
                bboxes = _get_person_bboxes(det_result, score_thr=0.3)

                frame_poses = []
                if len(bboxes) > 0:
                    pose_results = inference_topdown(pose_model, frame, bboxes)

                    for r in pose_results:
                        if hasattr(r, 'pred_instances'):
                            kpts = r.pred_instances.keypoints[0]  # (K, 2)
                            scores = r.pred_instances.keypoint_scores[0]  # (K,)
                        else:
                            # Older format, (K,3): x,y,score
                            kpts = r.keypoints[0][:, :2]
                            scores = r.keypoints[0][:, 2]

                        k = min(len(kpts), body_kpts)
                        pose_data = np.zeros((body_kpts, 3), dtype=np.float32)
                        pose_data[:k, :2] = kpts[:k]
                        pose_data[:k, 2] = scores[:k]
                        frame_poses.append(pose_data)

                if len(frame_poses) == 0:
                    frame_poses.append(np.zeros((body_kpts, 3), dtype=np.float32))

                all_poses.append(frame_poses)
                pbar.update(1)

        cap.release()

        # Pack into tensor [T, P, K, 3]
        max_persons = max(len(fp) for fp in all_poses) if all_poses else 1
        T = len(all_poses)
        pose_tensor = torch.zeros((T, max_persons, body_kpts, 3), dtype=torch.float32)

        for i, frame_poses in enumerate(all_poses):
            for j, pose in enumerate(frame_poses):
                pose_tensor[i, j] = torch.from_numpy(pose)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pose_tensor, output_path)
        return True

    except Exception as e:
        print(f"Error extracting pose features from {video_path}: {e}")
        import traceback;
        traceback.print_exc()
        return False


def extract_face_features(video_path, output_path, face_model, detector, face_kpts=98):
    """Extract face landmarks using person detector + upper-crop heuristic, then face head."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 0

        all_faces = []

        with tqdm(total=total_frames, desc="  Extracting face frames", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                det_result = inference_detector(detector, frame)
                bboxes = _get_person_bboxes(det_result, score_thr=0.3)

                frame_faces = []
                if len(bboxes) > 0:
                    # Estimate face boxes as upper portion of each person box
                    face_bboxes = []
                    for x1, y1, x2, y2 in bboxes:
                        h = max(0.0, y2 - y1)
                        w = max(0.0, x2 - x1)
                        if h <= 1 or w <= 1:
                            continue
                        # Use top ~45% height; expand slightly in width
                        fx1 = x1 - 0.05 * w
                        fx2 = x2 + 0.05 * w
                        fy1 = y1
                        fy2 = y1 + 0.45 * h
                        face_bboxes.append([fx1, fy1, fx2, fy2])

                    if face_bboxes:
                        face_bboxes = np.array(face_bboxes, dtype=np.float32)
                        face_results = inference_topdown(face_model, frame, face_bboxes)

                        for r in face_results:
                            if hasattr(r, 'pred_instances'):
                                kpts = r.pred_instances.keypoints[0]  # (K,2)
                                scores = r.pred_instances.keypoint_scores[0]  # (K,)
                            else:
                                kpts = r.keypoints[0][:, :2]
                                scores = r.keypoints[0][:, 2]

                            k = min(len(kpts), face_kpts)
                            face_data = np.zeros((face_kpts, 3), dtype=np.float32)
                            face_data[:k, :2] = kpts[:k]
                            face_data[:k, 2] = scores[:k]
                            frame_faces.append(face_data)

                if len(frame_faces) == 0:
                    frame_faces.append(np.zeros((face_kpts, 3), dtype=np.float32))

                all_faces.append(frame_faces)
                pbar.update(1)

        cap.release()

        # Pack into tensor [T, F, K, 3]
        max_faces = max(len(ff) for ff in all_faces) if all_faces else 1
        T = len(all_faces)
        face_tensor = torch.zeros((T, max_faces, face_kpts, 3), dtype=torch.float32)

        for i, ff in enumerate(all_faces):
            for j, arr in enumerate(ff):
                face_tensor[i, j] = torch.from_numpy(arr)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(face_tensor, output_path)
        return True

    except Exception as e:
        print(f"Error extracting face features from {video_path}: {e}")
        import traceback;
        traceback.print_exc()
        return False


# -------------------------------
# Model initialization
# -------------------------------

def init_models(device='cuda:0', repos_dir=None):
    """Initialize detector + high-accuracy body & face models using local configs.

    Args:
        device: Device for models (cuda:0 or cpu)
        repos_dir: Directory containing cloned mmpose and mmdetection repos
    """
    print("Initializing models...")

    # Setup repositories
    if repos_dir is None:
        repos_dir = Path.cwd()
    else:
        repos_dir = Path(repos_dir)

    mmpose_dir, mmdet_dir = setup_repos(repos_dir)

    # Local config paths
    det_config = str(mmdet_dir / 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py')
    pose_config = str(
        mmpose_dir / 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py')
    face_config = str(
        mmpose_dir / 'configs/face_2d_keypoint/topdown_heatmap/wflw/td-hm_hrnetv2-w18_awing-8xb64-60e_wflw-256x256.py')

    # Checkpoint URLs (these still need to be downloaded)
    det_ckpt_url = (
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
        'rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
    )
    pose_ckpt_url = (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/'
        'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
    )
    face_ckpt_url = (
        'https://download.openmmlab.com/mmpose/face/hrnetv2/'
        'hrnetv2_w18_wflw_256x256_awing-5af5055c_20211212.pth'
    )

    # Check if config files exist
    for cfg_path, name in [(det_config, 'Detector'), (pose_config, 'Pose'), (face_config, 'Face')]:
        if not Path(cfg_path).exists():
            print(f"Error: {name} config not found at {cfg_path}")
            print(f"Please ensure mmpose and mmdetection are properly cloned at {repos_dir}")
            sys.exit(1)

    print(f"Using local configs:")
    print(f"  Detector: {det_config}")
    print(f"  Pose: {pose_config}")
    print(f"  Face: {face_config}")

    # Build models (pose first -> registry safety in some setups)
    pose_model = init_pose_model(pose_config, pose_ckpt_url, device=device)
    face_model = init_pose_model(face_config, face_ckpt_url, device=device)

    detector = init_detector(det_config, det_ckpt_url, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Detect keypoint counts dynamically (for robust zero-filling & packing)
    body_k = _keypoint_count_from_model(pose_model, fallback=17)
    face_k = _keypoint_count_from_model(face_model, fallback=98)

    print(f"Models initialized successfully:")
    print(f"  Body keypoints: {body_k}")
    print(f"  Face keypoints: {face_k}")

    return detector, pose_model, face_model, body_k, face_k


# -------------------------------
# Annotation processing
# -------------------------------

def process_annotations(annotation_path, device='cuda:0', repos_dir=None):
    """Process all videos/audios listed in the JSONL annotation file."""

    base_dir = Path(annotation_path).parent

    # Init models
    detector, pose_model, face_model, body_k, face_k = init_models(
        device=device,
        repos_dir=repos_dir
    )

    # Read annotations
    annotations = []
    if annotation_path.endswith('.jsonl'):
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                annotations.append(json.loads(line))
    else:
        # find all videos and create a dummy annotation
        video_files = list(base_dir.glob('**/*.mp4')) + list(base_dir.glob('**/*.avi'))
        for video_file in video_files:
            annotations.append({
                'videos': [str(video_file.relative_to(base_dir))],
                'audios': [],
                'pose': None,
                'face': None,
                'opensmile': None
            })

    # Process each sample
    for ann in tqdm(annotations, desc="Processing annotations"):
        # Videos
        for video_path in ann.get('videos', []):
            full_video_path = base_dir / video_path
            if not full_video_path.exists():
                print(f"Video not found: {full_video_path}")
                continue

            # Pose
            pose_output = base_dir / 'pose' / f"{video_path}.pt"
            if not pose_output.exists():
                print(f"Extracting pose features for {video_path}")
                ok = extract_pose_features(full_video_path, pose_output, pose_model, detector, body_kpts=body_k)
                if ok:
                    ann['pose'] = str(Path('pose') / f"{video_path}.pt")
            else:
                ann['pose'] = str(Path('pose') / f"{video_path}.pt")

            # Face
            face_output = base_dir / 'face' / f"{video_path}.pt"
            if not face_output.exists():
                print(f"Extracting face features for {video_path}")
                ok = extract_face_features(full_video_path, face_output, face_model, detector, face_kpts=face_k)
                if ok:
                    ann['face'] = str(Path('face') / f"{video_path}.pt")
            else:
                ann['face'] = str(Path('face') / f"{video_path}.pt")

            # OpenSmile from video (audio track)
            opensmile_output = base_dir / 'opensmile' / f"{video_path}.pt"
            if not opensmile_output.exists():
                print(f"Extracting OpenSmile features for {video_path}")
                ok = extract_opensmile_features(full_video_path, opensmile_output)
                if ok:
                    ann['opensmile'] = str(Path('opensmile') / f"{video_path}.pt")
            else:
                ann['opensmile'] = str(Path('opensmile') / f"{video_path}.pt")

        # Audios
        for audio_path in ann.get('audios', []):
            full_audio_path = base_dir / audio_path
            if not full_audio_path.exists():
                print(f"Audio not found: {full_audio_path}")
                continue

            opensmile_output = base_dir / 'opensmile' / f"{audio_path}.pt"
            if not opensmile_output.exists():
                print(f"Extracting OpenSmile features for {audio_path}")
                ok = extract_opensmile_features(full_audio_path, opensmile_output)
                if ok:
                    ann['opensmile'] = str(Path('opensmile') / f"{audio_path}.pt")
            else:
                ann['opensmile'] = str(Path('opensmile') / f"{audio_path}.pt")

    # Save updated annotations
    output_path = str(annotation_path).replace('.jsonl', '_with_features.jsonl')
    with open(output_path, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')

    print(f"Updated annotations saved to {output_path}")


# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract OpenSmile, body pose, and face keypoints from clinical videos/audios')
    parser.add_argument('annotation_path', type=str, nargs='?',
                        default='/orcd/scratch/seedfund/001/multimodal/dvd/human_behaviour_data/train_template_prompts.jsonl',
                        help='Path to the annotation JSONL file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for models, e.g., cuda:0 or cpu')
    parser.add_argument('--repos-dir', type=str, default=None,
                        help='Directory to clone/find mmpose and mmdetection repos (default: current directory)')

    args = parser.parse_args()

    process_annotations(
        args.annotation_path,
        device=args.device,
        repos_dir=args.repos_dir
    )


if __name__ == '__main__':
    main()