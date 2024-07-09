import datetime
import glob
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import liblzfse
import numpy as np
from tqdm import tqdm

import stretch.utils.git_tools as git_tools

logger = logging.getLogger(__name__)

COMPLETION_FILENAME = "rgb_rel_videos_exported.txt"
IMG_COMPLETION_FILENAME = "completed.txt"
ABANDONED_FILENAME = "abandoned.txt"

RGB_VIDEO_H264_NAME = "gripper_compressed_video_h264.mp4"
HEAD_RGB_VIDEO_H264_NAME = "head_compressed_video_h264.mp4"

DEPTH_FOLDER_NAME = "compressed_gripper_depths"
RGB_FOLDER_NAME = "compressed_gripper_images"
HEAD_DEPTH_FOLDER_NAME = "compressed_head_depths"
HEAD_RGB_FOLDER_NAME = "compressed_head_images"

COMPLETED_DEPTH_FILENAME = "compressed_np_gripper_depth_float32.bin"
COMPLETED_HEAD_DEPTH_FILENAME = "compressed_np_head_depth_float32.bin"


class FileDataRecorder:
    """A class for writing out data to files for use in learning from demonstration. This one will create a folder structure with images and a text file containing position information."""

    def __init__(
        self,
        datadir: str = "./data",
        task: str = "default_task",
        user: str = "default_user",
        env: str = "default_env",
        save_images: bool = False,
        metadata: dict = None,
    ):
        """Initialize the recorder.

        Args:
            datadir: The directory to save the data in.
            task: The name of the task.
            user: The name of the user.
            env: The name of the environment.
        """
        if isinstance(datadir, Path):
            self.datadir = datadir
        else:
            self.datadir = Path(datadir)
        self.task_dir = self.datadir / task / user / env
        try:
            self.task_dir.mkdir(parents=True)
        except FileExistsError:
            pass
        self.save_images = save_images

        self.metadata = metadata

        self.reset()

    def reset(self):
        """Clear the data stored in the recorder."""
        self.rgbs = []
        self.depths = []
        self.head_rgbs = []
        self.head_depths = []
        self.data_dicts = {}
        self.step = 0

    def add(
        self,
        ee_rgb: np.ndarray,
        ee_depth: np.ndarray,
        xyz: np.ndarray,
        quaternion: np.ndarray,
        gripper: float,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        observations: Dict[str, float],
        actions: Dict[str, float],
        head_rgb: Optional[np.ndarray] = None,
        head_depth: Optional[np.ndarray] = None,
    ):
        """Add data to the recorder."""
        self.rgbs.append(ee_rgb)
        self.depths.append(ee_depth)
        self.head_rgbs.append(head_rgb)
        self.head_depths.append(head_depth)
        self.data_dicts[self.step] = {
            "xyz": xyz.tolist(),
            "quats": quaternion.tolist(),
            "gripper": gripper,
            "step": self.step,
            "ee_pos": ee_pos.tolist(),
            "ee_rot": ee_rot.tolist(),
            "observations": observations,
            "actions": actions,
        }
        self.step += 1

    def write(self, success: Optional[bool] = None):
        """Write out the data to a file."""

        now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

        # Create the episode directory
        episode_dir = self.task_dir / now
        episode_dir.mkdir()

        # Write the images
        print("Write end effector camera feed...")
        for i, (rgb, depth) in tqdm(enumerate(zip(self.rgbs, self.depths)), ncols=50):
            self.write_image(rgb, depth, episode_dir, i)

        print("Write head camera feed...")
        for i, (rgb, depth) in tqdm(enumerate(zip(self.head_rgbs, self.head_depths)), ncols=50):
            if rgb is None or depth is None:
                continue
            self.write_image(rgb, depth, episode_dir, i, head=True)

        # Run video processing
        print("Processing end effector camera feed...")
        self.process_rgb_to_video(episode_dir)
        self.process_depth_to_bin(episode_dir)
        print("Processing head camera feed...")
        self.process_rgb_to_video(episode_dir, head=True)
        self.process_depth_to_bin(episode_dir, head=True)

        print("Writing metadata...")

        # Bookkeeping for DobbE
        # Write an empty file
        with open(str(episode_dir / "rgb_rel_videos_exported.txt"), "w") as file:
            pass
        # Write a file saying this is done
        with open(str(episode_dir / "completed.txt"), "w") as file:
            # Write the string to the file
            file.write("Completed")

        # We only write success if it is explicitly provided
        if success is not None:
            # Write success or failure
            with open(str(episode_dir / "success.txt"), "w") as file:
                # Write the string to the file
                if success:
                    file.write("Success")
                else:
                    file.write("Failure")

        with open(episode_dir / "labels.json", "w") as f:
            json.dump(self.data_dicts, f)

        # Add episode info to metadata
        self.metadata["date"] = now
        self.metadata["num_frames"] = len(self.rgbs)

        # Collect git information if it exists
        self.metadata["git_branch"] = git_tools.get_git_branch()
        self.metadata["git_commit"] = git_tools.get_git_commit()
        self.metadata["git_commit_message"] = git_tools.get_git_commit_message()

        # Write metadata json file
        with open(str(episode_dir / "configs.json"), "w") as fp:
            json.dump(self.metadata, fp)

        if not self.save_images:
            self.cleanup_image_folders(episode_dir)

        # Reset the recorder
        self.reset()
        print("Done!")

    def write_image(self, rgb, depth, episode_dir, i, head: bool = False):
        """Write out image data from both head and end effector"""
        if head:
            rgb_dir = episode_dir / HEAD_RGB_FOLDER_NAME
            depth_dir = episode_dir / HEAD_DEPTH_FOLDER_NAME
        else:
            rgb_dir = episode_dir / RGB_FOLDER_NAME
            depth_dir = episode_dir / DEPTH_FOLDER_NAME

        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(rgb_dir / f"{i:06}.png"), rgb)
        cv2.imwrite(str(depth_dir / f"{i:06}.png"), depth)

    def cleanup_image_folders(self, episode_dir):
        """Delete image and depth folders when raw images and depths are no longer needed"""

        head_rgb_dir = episode_dir / HEAD_RGB_FOLDER_NAME
        head_depth_dir = episode_dir / HEAD_DEPTH_FOLDER_NAME
        rgb_dir = episode_dir / RGB_FOLDER_NAME
        depth_dir = episode_dir / DEPTH_FOLDER_NAME

        image_folders = [head_rgb_dir, head_depth_dir, rgb_dir, depth_dir]

        for folder in image_folders:
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder)

    def process_rgb_to_video(self, episode_dir, head: bool = False):
        start_time = time.perf_counter()
        # First, find out a sample filename
        if head:
            rgb_dir = episode_dir / HEAD_RGB_FOLDER_NAME
            h264_video_path = episode_dir / HEAD_RGB_VIDEO_H264_NAME
        else:
            rgb_dir = episode_dir / RGB_FOLDER_NAME
            h264_video_path = episode_dir / RGB_VIDEO_H264_NAME
        try:
            sample_filename = next(rgb_dir.glob("*.png"))
        except StopIteration:
            sample_filename = None
        if sample_filename is None:
            logging.error(f"No images found in {rgb_dir}")
            return

        # Find out if the filename is 4 or 6 digits long.
        if len(sample_filename.stem) == 4:
            filename_format = "%04d.png"
        elif len(sample_filename.stem) == 6:
            filename_format = "%06d.png"
        else:
            logging.error(f"Unknown filename format: {sample_filename.stem}")
            return

        # Now, we create the videos using ffmpeg.
        # First, we will create the h264 video.
        # Additional codecs can be output by adding to the list and providing corresponding video paths
        crfs = [30]
        video_codecs = ["h264"]
        for enc_lib, crf, final_video_path in zip(video_codecs, crfs, [h264_video_path]):
            command = [
                "ffmpeg",
                "-y",
                "-framerate",
                "15",
                "-i",
                str(rgb_dir / "{}").format(filename_format),
                "-c:v",
                enc_lib,
                "-crf",
                str(crf),
                str(final_video_path),
            ]
            process = subprocess.run(
                command,
                capture_output=True,
                check=True,
            )
            process.check_returncode()
            logging.info(process.stdout.decode("utf-8"))
            logging.debug(process.stderr.decode("utf-8"))

        end_time = time.perf_counter()
        logger.info(f"Saved RGB video to {episode_dir} in {end_time - start_time}s")

    def process_depth_to_bin(self, episode_dir: Path, head: bool = False) -> None:
        if head:
            all_depth_data = np.stack(self.head_depths, axis=0)
            target_depth_filename = episode_dir / COMPLETED_HEAD_DEPTH_FILENAME
        else:
            all_depth_data = np.stack(self.depths, axis=0)
            target_depth_filename = episode_dir / COMPLETED_DEPTH_FILENAME
        # Now zip and save this depth data.
        depth_array = all_depth_data
        depth_bytes = liblzfse.compress(depth_array.astype(np.float32).tobytes())
        target_depth_filename.write_bytes(depth_bytes)

        # TODO: remove debug code
        # This should be 192 x 256 x 4 bytes = 196608 bytes per image
        # buffer = np.frombuffer(
        #        liblzfse.decompress(target_depth_filename.read_bytes()), dtype=np.float32
        #   )
