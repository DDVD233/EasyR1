#!/usr/bin/env python3
"""
Feature extraction script for clinical videos/audios.
Extracts OpenSmile, pose, and face features and saves them as .pt files.
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

warnings.filterwarnings('ignore')

# OpenSmile
import opensmile

# MMPose and MMDetection
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.utils import adapt_mmdet_pipeline


def extract_opensmile_features(audio_path, output_path):
    """Extract OpenSmile features from audio/video file."""
    try:
        # Use ComParE_2016 feature set (comprehensive acoustic features)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Extract features
        features = smile.process_file(audio_path)

        # Convert to tensor and save
        feature_tensor = torch.tensor(features.values, dtype=torch.float32)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_tensor, output_path)

        return True
    except Exception as e:
        print(f"Error extracting OpenSmile features from {audio_path}: {e}")
        return False


def extract_pose_features(video_path, output_path, pose_model, detector):
    """Extract pose features from video using MMPose."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_poses = []

        # Create progress bar for frames
        with tqdm(total=total_frames, desc="  Extracting pose frames", leave=False) as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect persons
                det_result = inference_detector(detector, frame)

                # Get person bounding boxes (class 0 is person in COCO)
                # Handle different detector output formats
                if hasattr(det_result, 'pred_instances'):
                    pred_instances = det_result.pred_instances
                    bboxes = pred_instances.bboxes[pred_instances.labels == 0]
                    if hasattr(bboxes, 'cpu'):
                        bboxes = bboxes.cpu().numpy()
                else:
                    # Fallback for older MMDet versions
                    bboxes = det_result[0][0]  # First class (person) bboxes
                    bboxes = bboxes[bboxes[:, 4] > 0.3]  # Confidence threshold

                frame_poses = []

                if len(bboxes) > 0:
                    # Ensure bboxes shape is correct
                    if bboxes.shape[1] > 4:
                        bboxes = bboxes[:, :4]  # Take only x1, y1, x2, y2

                    # Run pose estimation for each person
                    pose_results = inference_topdown(pose_model, frame, bboxes)

                    for pose_result in pose_results:
                        # Extract keypoints (x, y, confidence)
                        if hasattr(pose_result, 'pred_instances'):
                            keypoints = pose_result.pred_instances.keypoints[0]  # Shape: (17, 2) for COCO
                            keypoint_scores = pose_result.pred_instances.keypoint_scores[0]  # Shape: (17,)
                        else:
                            # Fallback for different result format
                            keypoints = pose_result.keypoints[0][:, :2]
                            keypoint_scores = pose_result.keypoints[0][:, 2]

                        # Combine keypoints with scores
                        pose_data = np.concatenate([keypoints, keypoint_scores.reshape(-1, 1)], axis=1)
                        frame_poses.append(pose_data)

                # If no poses detected, append zeros
                if len(frame_poses) == 0:
                    frame_poses.append(np.zeros((17, 3)))  # 17 keypoints for COCO

                all_poses.append(frame_poses)
                pbar.update(1)

        cap.release()

        # Convert to tensor
        # Structure: [num_frames, max_persons_per_frame, num_keypoints, 3 (x, y, conf)]
        max_persons = max(len(frame_poses) for frame_poses in all_poses) if all_poses else 1

        pose_tensor = torch.zeros((len(all_poses), max_persons, 17, 3))
        for i, frame_poses in enumerate(all_poses):
            for j, pose in enumerate(frame_poses):
                pose_tensor[i, j] = torch.tensor(pose, dtype=torch.float32)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pose_tensor, output_path)

        return True
    except Exception as e:
        print(f"Error extracting pose features from {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_face_features(video_path, output_path, face_model, detector):
    """Extract face landmark features from video using MMPose."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_faces = []

        # Create progress bar for frames
        with tqdm(total=total_frames, desc="  Extracting face frames", leave=False) as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces/persons
                det_result = inference_detector(detector, frame)

                # Get person bounding boxes
                if hasattr(det_result, 'pred_instances'):
                    pred_instances = det_result.pred_instances
                    bboxes = pred_instances.bboxes[pred_instances.labels == 0]
                    if hasattr(bboxes, 'cpu'):
                        bboxes = bboxes.cpu().numpy()
                else:
                    bboxes = det_result[0][0]
                    bboxes = bboxes[bboxes[:, 4] > 0.3]

                frame_faces = []

                if len(bboxes) > 0:
                    # For each person bbox, estimate face region (upper 1/3 of person bbox)
                    face_bboxes = []
                    for bbox in bboxes:
                        if bbox.shape[0] >= 4:
                            x1, y1, x2, y2 = bbox[:4]
                            # Estimate face region as upper portion of person bbox
                            face_height = (y2 - y1) * 0.3
                            face_bbox = [x1, y1, x2, y1 + face_height]
                            face_bboxes.append(face_bbox)

                    if face_bboxes:
                        face_bboxes = np.array(face_bboxes)

                        # Run face landmark detection
                        face_results = inference_topdown(face_model, frame, face_bboxes)

                        for face_result in face_results:
                            # Extract keypoints
                            if hasattr(face_result, 'pred_instances'):
                                keypoints = face_result.pred_instances.keypoints[0]
                                keypoint_scores = face_result.pred_instances.keypoint_scores[0]
                            else:
                                keypoints = face_result.keypoints[0][:, :2]
                                keypoint_scores = face_result.keypoints[0][:, 2]

                            # Combine keypoints with scores
                            face_data = np.concatenate([keypoints, keypoint_scores.reshape(-1, 1)], axis=1)
                            frame_faces.append(face_data)

                # If no faces detected, append zeros (106 landmarks for face)
                if len(frame_faces) == 0:
                    num_landmarks = 106  # Default, will be adjusted based on model
                    frame_faces.append(np.zeros((num_landmarks, 3)))

                all_faces.append(frame_faces)
                pbar.update(1)

        cap.release()

        # Convert to tensor
        max_faces = max(len(frame_faces) for frame_faces in all_faces) if all_faces else 1

        # Get number of landmarks from first valid detection
        num_landmarks = 106
        for frame_faces in all_faces:
            if frame_faces and frame_faces[0].shape[0] > 0:
                num_landmarks = frame_faces[0].shape[0]
                break

        face_tensor = torch.zeros((len(all_faces), max_faces, num_landmarks, 3))
        for i, frame_faces in enumerate(all_faces):
            for j, face in enumerate(frame_faces):
                if face.shape[0] == num_landmarks:
                    face_tensor[i, j] = torch.tensor(face, dtype=torch.float32)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(face_tensor, output_path)

        return True
    except Exception as e:
        print(f"Error extracting face features from {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_config(url, filename):
    """Download config file from URL to a temporary location."""
    temp_dir = Path(tempfile.gettempdir()) / 'mmpose_configs'
    temp_dir.mkdir(exist_ok=True)

    config_path = temp_dir / filename

    # Download if not already cached
    if not config_path.exists():
        print(f"Downloading config: {filename}")
        urllib.request.urlretrieve(url, config_path)

    return str(config_path)


def init_models():
    """Initialize all models using model zoo URLs directly."""
    print("Initializing models...")

    # Download config files from GitHub
    detector_config = download_config(
        'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py',
        'rtmdet_m_640-8xb32_coco-person.py'
    )

    pose_config = download_config(
        'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py',
        'rtmpose-l_8xb256-420e_coco-256x192.py'
    )

    face_config = download_config(
        'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/face_2d_keypoint/rtmpose-m_8xb256-120e_lapa-256x256.py',
        'rtmpose-m_8xb256-120e_lapa-256x256.py'
    )

    # Initialize models with local configs and CDN checkpoints
    # Initialize pose models first to avoid registry issues
    pose_model = init_pose_model(
        pose_config,
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth',
        device='cuda:0'
    )

    face_model = init_pose_model(
        face_config,
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth',
        device='cuda:0'
    )

    # Initialize detector last
    detector = init_detector(
        detector_config,
        'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        device='cuda:0'
    )

    # Fix the detector pipeline for use in mmpose scope
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    return detector, pose_model, face_model


def process_annotations(annotation_path):
    """Process all videos/audios in the annotation file."""

    # Setup paths
    base_dir = Path(annotation_path).parent

    # Initialize models
    try:
        detector, pose_model, face_model = init_models()
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Make sure you have mmpose and mmdet installed properly")
        return

    # Read annotations
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))

    # Process each annotation
    for ann in tqdm(annotations, desc="Processing annotations"):
        # Process videos
        for video_path in ann.get('videos', []):
            full_video_path = base_dir / video_path

            if not full_video_path.exists():
                print(f"Video not found: {full_video_path}")
                continue

            # Extract pose features
            pose_output = base_dir / 'pose' / f"{video_path}.pt"
            if not pose_output.exists():
                print(f"Extracting pose features for {video_path}")
                if extract_pose_features(full_video_path, pose_output, pose_model, detector):
                    ann['pose'] = str(Path('pose') / f"{video_path}.pt")
            else:
                ann['pose'] = str(Path('pose') / f"{video_path}.pt")

            # Extract face features
            face_output = base_dir / 'face' / f"{video_path}.pt"
            if not face_output.exists():
                print(f"Extracting face features for {video_path}")
                if extract_face_features(full_video_path, face_output, face_model, detector):
                    ann['face'] = str(Path('face') / f"{video_path}.pt")
            else:
                ann['face'] = str(Path('face') / f"{video_path}.pt")

            # Extract OpenSmile features from video audio
            opensmile_output = base_dir / 'opensmile' / f"{video_path}.pt"
            if not opensmile_output.exists():
                print(f"Extracting OpenSmile features for {video_path}")
                if extract_opensmile_features(full_video_path, opensmile_output):
                    ann['opensmile'] = str(Path('opensmile') / f"{video_path}.pt")
            else:
                ann['opensmile'] = str(Path('opensmile') / f"{video_path}.pt")

        # Process audios
        for audio_path in ann.get('audios', []):
            full_audio_path = base_dir / audio_path

            if not full_audio_path.exists():
                print(f"Audio not found: {full_audio_path}")
                continue

            # Extract OpenSmile features
            opensmile_output = base_dir / 'opensmile' / f"{audio_path}.pt"
            if not opensmile_output.exists():
                print(f"Extracting OpenSmile features for {audio_path}")
                if extract_opensmile_features(full_audio_path, opensmile_output):
                    ann['opensmile'] = str(Path('opensmile') / f"{audio_path}.pt")
            else:
                ann['opensmile'] = str(Path('opensmile') / f"{audio_path}.pt")

    # Save updated annotations
    output_path = annotation_path.replace('.jsonl', '_with_features.jsonl')
    with open(output_path, 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')

    print(f"Updated annotations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from clinical videos/audios')
    parser.add_argument(
        'annotation_path',
        type=str,
        nargs='?',
        default='/orcd/scratch/seedfund/001/multimodal/dvd/human_behaviour_data/train_template_prompts.jsonl',
        help='Path to the annotation JSONL file'
    )

    args = parser.parse_args()

    if not os.path.exists(args.annotation_path):
        print(f"Error: Annotation file not found: {args.annotation_path}")
        return

    process_annotations(args.annotation_path)


if __name__ == '__main__':
    main()