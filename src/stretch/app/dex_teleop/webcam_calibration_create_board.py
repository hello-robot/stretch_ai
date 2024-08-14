# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import cv2
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
