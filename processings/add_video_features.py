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

This script is designed to be drop-in compatible with the user's prior script but
with stronger models by default. It also adapts automatically to the model's
declared number of keypoints, so if you swap configs/weights, shapes update safely.
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
import urllib.request
import tempfile
import hashlib

warnings.filterwarnings('ignore')

# OpenSmile
import opensmile

# MMPose and MMDetection
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.utils import adapt_mmdet_pipeline


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
                            kpts = r.pred_instances.keypoints[0]         # (K, 2)
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
        import traceback; traceback.print_exc()
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
                                kpts = r.pred_instances.keypoints[0]         # (K,2)
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
        import traceback; traceback.print_exc()
        return False


# -------------------------------
# Model initialization
# -------------------------------

def download_config(url, filename=None):
    """Download a config file to a temp cache, namespaced by URL."""
    temp_dir = Path(tempfile.gettempdir()) / 'mmpose_configs'
    temp_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        digest = hashlib.md5(url.encode()).hexdigest()
        filename = f'{digest}.py'

    config_path = temp_dir / filename
    if not config_path.exists():
        print(f"Downloading config: {filename}")
        urllib.request.urlretrieve(url, config_path)

    return str(config_path)


def init_models(device='cuda:0',
                det_config_url=None, det_ckpt_url=None,
                pose_config_url=None, pose_ckpt_url=None,
                face_config_url=None, face_ckpt_url=None):
    """Initialize detector + high-accuracy body & face models.

    You can override any of the URLs above via CLI flags if the defaults change.
    """
    print("Initializing models...")

    # Defaults (picked for accuracy)
    det_config_url = det_config_url or (
        'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/'
        'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
    )
    det_ckpt_url = det_ckpt_url or (
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
        'rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
    )

    pose_config_url = pose_config_url or (
        'https://raw.githubusercontent.com/open-mmlab/mmpose/dev-1.x/'
        'configs/body_2d_keypoint/topdown_heatmap/coco/'
        'td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
    )
    pose_ckpt_url = pose_ckpt_url or (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/'
        'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
    )

    face_config_url = face_config_url or (
        'https://raw.githubusercontent.com/open-mmlab/mmpose/dev-1.x/'
        'configs/face_2d_keypoint/topdown_heatmap/wflw/'
        'td-hm_hrnetv2-w18_awing-8xb64-60e_wflw-256x256.py'
    )
    face_ckpt_url = face_ckpt_url or (
        'https://download.openmmlab.com/mmpose/face/hrnetv2/'
        'hrnetv2_w18_wflw_256x256_awing-5af5055c_20211212.pth'
    )

    # Download configs
    det_cfg = download_config(det_config_url, 'rtmdet_l_8xb32-300e_coco.py')
    pose_cfg = download_config(pose_config_url, 'td-hm_ViTPose-huge_256x192.py')
    face_cfg = download_config(face_config_url, 'td-hm_hrnetv2-w18_awing_wflw_256x256.py')

    # Build models (pose first -> registry safety in some setups)
    pose_model = init_pose_model(pose_cfg, pose_ckpt_url, device=device)
    face_model = init_pose_model(face_cfg, face_ckpt_url, device=device)

    detector = init_detector(det_cfg, det_ckpt_url, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Detect keypoint counts dynamically (for robust zero-filling & packing)
    body_k = _keypoint_count_from_model(pose_model, fallback=17)
    face_k = _keypoint_count_from_model(face_model, fallback=98)

    return detector, pose_model, face_model, body_k, face_k


# -------------------------------
# Annotation processing
# -------------------------------

def process_annotations(annotation_path,
                        device='cuda:0',
                        det_config_url=None, det_ckpt_url=None,
                        pose_config_url=None, pose_ckpt_url=None,
                        face_config_url=None, face_ckpt_url=None):
    """Process all videos/audios listed in the JSONL annotation file."""

    base_dir = Path(annotation_path).parent

    # Init models
    try:
        detector, pose_model, face_model, body_k, face_k = init_models(
            device=device,
            det_config_url=det_config_url, det_ckpt_url=det_ckpt_url,
            pose_config_url=pose_config_url, pose_ckpt_url=pose_ckpt_url,
            face_config_url=face_config_url, face_ckpt_url=face_ckpt_url
        )
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Make sure mmpose==1.x and mmdetection are installed correctly, and CUDA is available if using cuda device.")
        return

    # Read annotations
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            annotations.append(json.loads(line))

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
    parser = argparse.ArgumentParser(description='Extract OpenSmile, body pose, and face keypoints from clinical videos/audios')
    parser.add_argument('annotation_path', type=str, nargs='?',
                        default='/orcd/scratch/seedfund/001/multimodal/dvd/human_behaviour_data/train_template_prompts.jsonl',
                        help='Path to the annotation JSONL file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for models, e.g., cuda:0 or cpu')

    # Optional overrides for configs/checkpoints (if URLs change)
    parser.add_argument('--det-config-url', type=str, default=None)
    parser.add_argument('--det-ckpt-url', type=str, default=None)
    parser.add_argument('--pose-config-url', type=str, default=None)
    parser.add_argument('--pose-ckpt-url', type=str, default=None)
    parser.add_argument('--face-config-url', type=str, default=None)
    parser.add_argument('--face-ckpt-url', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.annotation_path):
        print(f"Error: Annotation file not found: {args.annotation_path}")
        return

    process_annotations(
        args.annotation_path,
        device=args.device,
        det_config_url=args.det_config_url, det_ckpt_url=args.det_ckpt_url,
        pose_config_url=args.pose_config_url, pose_ckpt_url=args.pose_ckpt_url,
        face_config_url=args.face_config_url, face_ckpt_url=args.face_ckpt_url
    )


if __name__ == '__main__':
    main()
