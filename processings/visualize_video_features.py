"""
Visualization script for extracted pose and face keypoints.
Reads .pt files and overlays keypoints on original videos.
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

warnings.filterwarnings('ignore')

# COCO pose keypoint connections (17 keypoints)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Face
    [6, 12], [7, 13], [6, 7],  # Torso
    [6, 8], [7, 9], [8, 10], [9, 11],  # Arms
    [2, 3], [1, 2], [1, 3],  # Head
    [2, 4], [3, 5], [4, 6], [5, 7]  # Head to torso
]


# LAPA dataset face landmark connections (106 landmarks)
# Based on https://github.com/jd-opensource/lapa-dataset
def get_lapa_face_connections():
    connections = []

    # Face contour (0-32)
    for i in range(32):
        connections.append([i, i + 1])
    connections.append([32, 0])  # Close contour

    # Left eyebrow (33-41)
    for i in range(33, 41):
        connections.append([i, i + 1])

    # Right eyebrow (42-50)
    for i in range(42, 50):
        connections.append([i, i + 1])

    # Nose bridge (51-54)
    for i in range(51, 54):
        connections.append([i, i + 1])

    # Nose lower part (55-65)
    # Left nostril
    for i in range(55, 58):
        connections.append([i, i + 1])
    connections.append([58, 55])  # Close left nostril

    # Right nostril
    for i in range(59, 62):
        connections.append([i, i + 1])
    connections.append([62, 59])  # Close right nostril

    # Nose tip
    connections.append([63, 64])
    connections.append([64, 65])
    connections.append([63, 55])
    connections.append([65, 59])

    # Left eye (66-71)
    for i in range(66, 71):
        connections.append([i, i + 1])
    connections.append([71, 66])  # Close eye

    # Right eye (72-77)
    for i in range(72, 77):
        connections.append([i, i + 1])
    connections.append([77, 72])  # Close eye

    # Left pupil (96-97 are centers, skip connections)

    # Right pupil (98-99 are centers, skip connections)

    # Outer mouth (78-89)
    for i in range(78, 89):
        connections.append([i, i + 1])
    connections.append([89, 78])  # Close outer mouth

    # Inner mouth (90-95)
    for i in range(90, 95):
        connections.append([i, i + 1])
    connections.append([95, 90])  # Close inner mouth

    return connections


FACE_CONNECTIONS = get_lapa_face_connections()


def draw_pose_keypoints(frame, pose_data, color=(0, 255, 0), thickness=2, confidence_threshold=0.3):
    """
    Draw pose keypoints and skeleton on frame.

    Args:
        frame: Video frame
        pose_data: Tensor of shape [num_persons, 17, 3] (x, y, confidence)
        color: Color for keypoints and skeleton
        thickness: Line thickness
        confidence_threshold: Minimum confidence to draw keypoint
    """
    annotated_frame = frame.copy()

    for person_idx in range(pose_data.shape[0]):
        person_pose = pose_data[person_idx]

        # Skip if all keypoints have low confidence (likely no person)
        if torch.all(person_pose[:, 2] < confidence_threshold):
            continue

        # Draw skeleton connections
        for connection in COCO_SKELETON:
            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-indexed

            if kpt1_idx >= len(person_pose) or kpt2_idx >= len(person_pose):
                continue

            kpt1 = person_pose[kpt1_idx]
            kpt2 = person_pose[kpt2_idx]

            # Only draw if both keypoints have sufficient confidence
            if kpt1[2] > confidence_threshold and kpt2[2] > confidence_threshold:
                pt1 = (int(kpt1[0]), int(kpt1[1]))
                pt2 = (int(kpt2[0]), int(kpt2[1]))
                cv2.line(annotated_frame, pt1, pt2, color, thickness)

        # Draw keypoints
        for kpt_idx, kpt in enumerate(person_pose):
            if kpt[2] > confidence_threshold:  # Confidence threshold
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(annotated_frame, (x, y), 4, color, -1)
                cv2.circle(annotated_frame, (x, y), 5, (255, 255, 255), 1)  # White border

    return annotated_frame


def draw_face_keypoints(frame, face_data, color=(255, 0, 255), thickness=1, confidence_threshold=0.3):
    """
    Draw face keypoints and connections on frame.

    Args:
        frame: Video frame
        face_data: Tensor of shape [num_faces, num_landmarks, 3] (x, y, confidence)
        color: Color for keypoints and connections
        thickness: Line thickness
        confidence_threshold: Minimum confidence to draw keypoint
    """
    annotated_frame = frame.copy()

    for face_idx in range(face_data.shape[0]):
        face_landmarks = face_data[face_idx]

        # Skip if all keypoints have low confidence (likely no face)
        if torch.all(face_landmarks[:, 2] < confidence_threshold):
            continue

        # Draw connections for LAPA 106 landmarks
        for connection in FACE_CONNECTIONS:
            if connection[0] < len(face_landmarks) and connection[1] < len(face_landmarks):
                kpt1 = face_landmarks[connection[0]]
                kpt2 = face_landmarks[connection[1]]

                # Only draw if both keypoints have sufficient confidence
                if kpt1[2] > confidence_threshold and kpt2[2] > confidence_threshold:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(annotated_frame, pt1, pt2, color, thickness)

        # Draw keypoints
        for kpt_idx, kpt in enumerate(face_landmarks):
            if kpt[2] > confidence_threshold:  # Confidence threshold
                x, y = int(kpt[0]), int(kpt[1])
                # Special markers for pupils (landmarks 96-99 in LAPA)
                if 96 <= kpt_idx <= 99:
                    cv2.circle(annotated_frame, (x, y), 3, (0, 0, 255), -1)  # Red for pupils
                else:
                    cv2.circle(annotated_frame, (x, y), 2, color, -1)

    return annotated_frame


def visualize_video_with_keypoints(video_path, pose_pt_path=None, face_pt_path=None,
                                   output_path=None, show_pose=True, show_face=True,
                                   fps_reduction=1, confidence_threshold=0.3):
    """
    Create visualization video with keypoints overlaid.

    Args:
        video_path: Path to original video
        pose_pt_path: Path to pose .pt file
        face_pt_path: Path to face .pt file
        output_path: Path for output video
        show_pose: Whether to show pose keypoints
        show_face: Whether to show face keypoints
        fps_reduction: Factor to reduce FPS (1 = original, 2 = half, etc.)
        confidence_threshold: Minimum confidence to draw keypoints
    """
    # Load keypoint data
    pose_data = None
    face_data = None

    if show_pose and pose_pt_path and pose_pt_path.exists():
        try:
            pose_data = torch.load(pose_pt_path, map_location='cpu')
            print(f"  Loaded pose data: {pose_data.shape}")
        except Exception as e:
            print(f"  Error loading pose data: {e}")

    if show_face and face_pt_path and face_pt_path.exists():
        try:
            face_data = torch.load(face_pt_path, map_location='cpu')
            print(f"  Loaded face data: {face_data.shape}")
        except Exception as e:
            print(f"  Error loading face data: {e}")

    if pose_data is None and face_data is None:
        print(f"  No keypoint data found for {video_path}")
        return False

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error opening video: {video_path}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) / fps_reduction
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frames
    frame_idx = 0
    with tqdm(total=total_frames, desc="  Processing frames", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames based on fps_reduction
            if frame_idx % fps_reduction != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            adjusted_idx = frame_idx // fps_reduction

            # Draw pose keypoints
            if pose_data is not None and adjusted_idx < pose_data.shape[0]:
                frame = draw_pose_keypoints(frame, pose_data[adjusted_idx],
                                            color=(0, 255, 0), thickness=2,
                                            confidence_threshold=confidence_threshold)

            # Draw face keypoints
            if face_data is not None and adjusted_idx < face_data.shape[0]:
                frame = draw_face_keypoints(frame, face_data[adjusted_idx],
                                            color=(255, 0, 255), thickness=1,
                                            confidence_threshold=confidence_threshold)

            # Add frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | Conf threshold: {confidence_threshold}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add legend
            legend_y = 60
            if pose_data is not None:
                cv2.putText(frame, "Pose", (10, legend_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                legend_y += 25
            if face_data is not None:
                cv2.putText(frame, "Face", (10, legend_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, "Pupils", (10, legend_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"  Saved visualization to {output_path}")
    return True


def process_annotations(annotation_path, output_dir, max_videos=None,
                        show_pose=True, show_face=True, fps_reduction=1,
                        confidence_threshold=0.3):
    """
    Process all videos in the annotation file and create visualizations.

    Args:
        annotation_path: Path to annotation JSONL file
        output_dir: Directory for output visualizations
        max_videos: Maximum number of videos to process (None for all)
        show_pose: Whether to show pose keypoints
        show_face: Whether to show face keypoints
        fps_reduction: Factor to reduce FPS
        confidence_threshold: Minimum confidence to draw keypoints
    """
    # Setup paths
    base_dir = Path(annotation_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            })

    # Process videos
    video_count = 0
    for ann in tqdm(annotations, desc="Processing annotations"):
        for video_path in ann.get('videos', []):
            if max_videos and video_count >= max_videos:
                print(f"\nReached maximum of {max_videos} videos")
                return

            full_video_path = base_dir / video_path
            if not full_video_path.exists():
                print(f"\nVideo not found: {full_video_path}")
                continue

            # Check for existing feature files
            pose_pt_path = base_dir / 'pose' / f"{video_path}.pt"
            face_pt_path = base_dir / 'face' / f"{video_path}.pt"

            # Check if at least one feature file exists
            if not pose_pt_path.exists() and not face_pt_path.exists():
                print(f"\nNo feature files found for {video_path}")
                continue

            # Create output filename
            video_name = Path(video_path).stem
            output_path = output_dir / f"{video_name}_keypoints.mp4"

            # Skip if already processed
            if output_path.exists():
                print(f"\nVisualization already exists: {output_path}")
                video_count += 1
                continue

            print(f"\nProcessing video: {video_path}")

            # Create visualization
            success = visualize_video_with_keypoints(
                full_video_path,
                pose_pt_path if pose_pt_path.exists() else None,
                face_pt_path if face_pt_path.exists() else None,
                output_path,
                show_pose=show_pose,
                show_face=show_face,
                fps_reduction=fps_reduction,
                confidence_threshold=confidence_threshold
            )

            if success:
                video_count += 1

    print(f"\nProcessed {video_count} videos")
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize extracted keypoints on videos')
    parser.add_argument(
        'annotation_path',
        type=str,
        nargs='?',
        default='/orcd/scratch/seedfund/001/multimodal/dvd/human_behaviour_data/train_template_prompts.jsonl',
        help='Path to the annotation JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/orcd/scratch/seedfund/001/multimodal/dvd/visualizations',
        help='Directory for output visualizations'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process (default: all)'
    )
    parser.add_argument(
        '--no-pose',
        action='store_true',
        help='Do not show pose keypoints'
    )
    parser.add_argument(
        '--no-face',
        action='store_true',
        help='Do not show face keypoints'
    )
    parser.add_argument(
        '--fps-reduction',
        type=int,
        default=1,
        help='Factor to reduce FPS (1=original, 2=half, etc.)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Minimum confidence threshold for displaying keypoints (0.0-1.0, default: 0.3)'
    )

    args = parser.parse_args()

    process_annotations(
        args.annotation_path,
        args.output_dir,
        max_videos=args.max_videos,
        show_pose=not args.no_pose,
        show_face=not args.no_face,
        fps_reduction=args.fps_reduction,
        confidence_threshold=args.confidence_threshold
    )


if __name__ == '__main__':
    main()