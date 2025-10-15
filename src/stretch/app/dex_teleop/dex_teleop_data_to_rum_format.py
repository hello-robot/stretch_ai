# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import argparse
import json
import shutil
from pathlib import Path

import cv2

# Constants for your fixed filenames
GRIP_FILENAME = "gripper_compressed_video_h264.mp4"
HEAD_FILENAME = "head_compressed_video_h264.mp4"
OUT_FILENAME = "compressed_video_h264.mp4"
LOG_FILENAME = "r3d_files.txt"

# TODO: add use depth option, Ideally we should copy depth corresponding to the rgb video into compressed np_depth_float32.bin

DEPTH_FILENAME = "compressed_np_depth_float32.bin"


def process_folder(src: Path, dst: Path, mode: str, use_depth: bool):

    # Define video paths
    grip_vid = src / GRIP_FILENAME
    head_vid = src / HEAD_FILENAME
    cap = cv2.VideoCapture(str(grip_vid))
    if not (grip_vid.exists() and head_vid.exists()):
        print(f"  ⚠️ Missing expected video(s) in {src}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Compute duration in seconds
    duration_seconds = total_frames / fps if fps > 0 else 0

    if duration_seconds > 140:
        print(f"  ⚠️ {src} is too long ({duration_seconds:.1f}s)")
        return

    print(f"  ▶️ Writing to {dst} @ {fps:.1f} FPS")

    dst.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f"  ⚠️ {dst} does not exist")
    with open(dst / "rgb_rel_videos_exported.txt", "w") as f:
        f.write("Completed")

    # Copy labels.json
    labels_src = src / "labels.json"
    if labels_src.exists():
        shutil.copy(labels_src, dst / "labels.json")
    else:
        print(f"  ⚠️ labels.json not found in {src}")

    # Copy gripper video
    # Mode: concatenate, gripper, head
    # Concatenate: concatenate gripper and head videos together into a new one
    # Gripper: copy gripper video only
    # Head: copy head video only
    if mode == "gripper":
        if grip_vid.exists():
            shutil.copy(grip_vid, dst / OUT_FILENAME)
    elif mode == "head":
        if head_vid.exists():
            shutil.copy(head_vid, dst / OUT_FILENAME)
    else:
        # Open captures
        cap1 = cv2.VideoCapture(str(grip_vid))
        cap2 = cv2.VideoCapture(str(head_vid))
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        fps = min(fps1 or fps2, fps2 or fps1)

        # Setup writer: two 256×256 frames side by side → (512×256)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = dst / OUT_FILENAME
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (256 * 2, 256))

        print(f"  ▶️ Writing to {out_path} @ {fps:.1f} FPS")
        while True:
            ret1, f1 = cap1.read()
            ret2, f2 = cap2.read()
            if not (ret1 and ret2):
                break
            f1 = cv2.resize(f1, (256, 256))
            f2 = cv2.resize(f2, (256, 256))
            # concatenate horizontally
            out.write(cv2.hconcat([f1, f2]))
        cap1.release()
        cap2.release()
        out.release()

    print(f"  ✅ Done.")


# process_folder(Path('/home/hello-robot/peiqi/stretch_dex_teleop/data/hat_pick/default_user/default_env/2025-07-30--12-14-45'), Path('/home/hello-robot/peiqi/stretch_dex_teleop/data/test/processed'))


def main():
    parser = argparse.ArgumentParser(
        description="Resample & concatenate fixed robot camera videos (256×256), drop failures."
    )
    parser.add_argument(
        "--input_root", type=Path, help="Root folder containing raw '*/*/*/date/' data"
    )
    parser.add_argument("--task_name", type=str, default="default_task", help="Task name")
    parser.add_argument(
        "--mode",
        default="concatenate",
        choices=["concatenate", "gripper", "head"],
        help="Mode of operation",
    )
    parser.add_argument("--use_depth", action="store_true", help="Use depth camera")
    args = parser.parse_args()

    input_root = args.input_root / args.task_name
    output_root = args.input_root / (args.task_name + "_rum")
    output_root.mkdir(parents=True, exist_ok=True)

    print(args.mode, args.use_depth)

    # Path to save r3d.txt
    log_path = args.input_root / LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    folder_list = []

    for success_file in input_root.rglob("success.txt"):
        status = success_file.read_text().strip().lower()
        if status != "success":
            continue  # skip failures

        src_folder = success_file.parent
        rel = src_folder.relative_to(input_root)
        dst_folder = output_root / rel

        print(f"\nProcessing: {src_folder} → {dst_folder}")
        process_folder(src_folder, dst_folder, args.mode, args.use_depth)
        folder_list.append(str(src_folder.resolve()) + ".zip")

    with open(log_path, "a") as f:
        json.dump(folder_list, f)
        f.write("\n")


if __name__ == "__main__":
    main()
