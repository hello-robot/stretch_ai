# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

import cv2
import cv2.aruco as aruco
import webcam_calibration_aruco_board as ab

aruco_dict = ab.aruco_dict
board = ab.board

########
# From
# https://papersizes.online/paper-size/letter/
#
# "Letter size in pixels when using 600 DPI: 6600 x 5100 pixels."
########

########
# From
# https://docs.opencv.org/4.8.0/d4/db2/classcv_1_1aruco_1_1Board.html
#
# Parameters
# outSize	size of the output image in pixels.
# img	        output image with the board. The size of this image will be outSize and the board will be on the center, keeping the board proportions.
# marginSize	minimum margins (in pixels) of the board in the output image
# borderBits	width of the marker borders.
########

image_size = (5100, 6600)
margin_size = int(image_size[1] / 20)
border_bits = 1

board_image = board.generateImage(
    outSize=image_size, marginSize=margin_size, borderBits=border_bits
)

cv2.imwrite("webcam_aruco_calibration_board.png", board_image)
