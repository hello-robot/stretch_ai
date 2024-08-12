# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io

import cv2
import h5py
import imageio
import numpy as np
from PIL import Image
from pygifsicle import optimize
from tqdm import tqdm


def img_from_bytes(data: bytes, height=None, width=None, format="png") -> np.ndarray:
    """Convert image from png bytes"""
    image = Image.open(io.BytesIO(data), mode="r", formats=[format])
    # TODO: decide if default image format should switch over to webp
    # Issue: not quite as good at handling depth
    # image = Image.open(data, mode='r', formats=['webp'])
    if height and width:
        # mypy: this can be an ImageFile apparently?
        image = image.resize((width, height))  # type: ignore
    return np.asarray(image)


def pil_to_bytes(img: Image.Image, format="png") -> bytes:
    """Convert image to bytes using PIL"""
    data = io.BytesIO()
    img.save(data, format=format)
    return data.getvalue()


def img_to_bytes(img: np.ndarray, format="png") -> bytes:
    """Convert image to bytes"""
    pil_img = Image.fromarray(img)
    return pil_to_bytes(pil_img, format)


def torch_to_bytes(img: np.ndarray) -> bytes:
    """convert from channels-first image (torch) to bytes)"""
    assert len(img.shape) == 3
    img = np.rollaxis(img, 0, 3)
    return img_to_bytes(img)


def png_to_gif(group: h5py.Group, key: str, name: str, save=True, height=None, width=None):
    """
    Write key out as a gif
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted([(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]),
        ncols=50,
    ):
        bindata = img_stream[k][()]
        img = img_from_bytes(bindata, height, width)
        gif.append(img)
    if save:
        # Mypy note: the type of gif is List[np.ndarray] which is fine for this function.
        imageio.mimsave(name, gif)  # type: ignore
    else:
        return gif


def pngs_to_gifs(filename: str, key: str):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_gif(group, key, group_name + ".gif")


def schema_to_gifs(filename: str):
    keys = [
        "top_rgb",
        "right_rgb",
        "left_rgb",
        "wrist_rgb",
    ]
    h5 = h5py.File(filename, "r")
    x = 1
    for group_name, grp in h5.items():
        print(f"Processing {group_name}, {x}/{len(h5.keys())}")
        x += 1
        gifs = []
        gif_name = group_name + ".gif"
        for key in keys:
            if key in grp.keys():
                gifs.append(png_to_gif(grp, key, name="", height=120, width=155, save=False))
        # TODO logic for concatenating the gifs and saving with group's name
        concatenated_gif = None
        for gif in gifs:
            if gif:
                if concatenated_gif is not None:
                    concatenated_gif = np.hstack((concatenated_gif, gif))
                else:
                    concatenated_gif = gif
        imageio.mimsave(gif_name, concatenated_gif)
        optimize(gif_name)


def png_to_mp4(group: h5py.Group, key: str, name: str, fps=10):
    """
    Write key out as a mpt
    """
    print("Writing gif to file:", name)
    img_stream = group[key]
    writer = None

    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted([(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]),
        ncols=50,
    ):

        bindata = img_stream[k][()]
        _img = img_from_bytes(bindata)
        w, h = _img.shape[:2]
        img = np.zeros_like(_img)
        img[:, :, 0] = _img[:, :, 2]
        img[:, :, 1] = _img[:, :, 1]
        img[:, :, 2] = _img[:, :, 0]

        if writer is None:
            # Mypy note: this definitely does exist but mypy says it does not. Ignore error.
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")  # type: ignore
            writer = cv2.VideoWriter(name, fourcc, fps, (h, w))
        writer.write(img)
    writer.release()


def pngs_to_mp4(filename: str, key: str, vid_name: str, fps: int):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_mp4(group, key, str(vid_name) + "_" + group_name + ".mp4", fps=fps)
