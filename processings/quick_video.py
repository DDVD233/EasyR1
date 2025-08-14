import os
import cv2
import mediapipe as mp
from pathlib import Path
import argparse

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def process_video(input_path, output_path):
    """Process a single video file with pose and face keypoints overlay."""

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Total frames: {total_frames}")

    # Buffer to store processed frames
    frame_buffer = []
    SAVE_INTERVAL = 1000  # Save every 1000 frames

    # Initialize pose and face detection
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, \
            mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processing frame {frame_count}/{total_frames}...")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Process pose and face
            pose_results = pose.process(image_rgb)
            face_results = face_mesh.process(image_rgb)

            # Convert back to BGR for drawing
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Draw face landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # Add frame to buffer
            frame_buffer.append(image_bgr)

            # Save partial video every SAVE_INTERVAL frames
            if frame_count % SAVE_INTERVAL == 0:
                print(f"  Saving partial video at frame {frame_count}...")
                save_video_buffer(frame_buffer, output_path, fps, width, height)

    # Save any remaining frames
    if frame_buffer:
        print(f"  Saving final video with {frame_count} frames...")
        save_video_buffer(frame_buffer, output_path, fps, width, height)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"  Completed: {output_path}")


def save_video_buffer(frame_buffer, output_path, fps, width, height):
    """Save the accumulated frame buffer to video file, overwriting if exists."""

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Write all frames
    for frame in frame_buffer:
        out.write(frame)

    # Release writer
    out.release()


def traverse_and_process(directory):
    """Traverse directory for .mp4 files and process them."""

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    # Find all .mp4 files recursively
    mp4_files = list(dir_path.rglob("*.mp4"))

    # Filter out already processed files
    mp4_files = [f for f in mp4_files if not f.stem.endswith("_keypoints")]

    if not mp4_files:
        print("No .mp4 files found to process")
        return

    print(f"Found {len(mp4_files)} .mp4 file(s) to process")

    for i, mp4_file in enumerate(mp4_files, 1):
        print(f"\n[{i}/{len(mp4_files)}] Processing: {mp4_file}")

        # Create output filename in the same directory
        output_file = mp4_file.parent / f"{mp4_file.stem}_keypoints.mp4"

        try:
            process_video(mp4_file, output_file)
        except Exception as e:
            print(f"  Error processing {mp4_file}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Generate body and face keypoints overlay for MP4 videos")
    parser.add_argument("directory", help="Directory to search for .mp4 files")
    args = parser.parse_args()

    traverse_and_process(args.directory)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()