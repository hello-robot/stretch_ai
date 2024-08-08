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

#
# References used when writing this code for OpenCV 4.8
#
# https://docs.opencv.org/4.8.0/df/d4a/tutorial_charuco_detection.html
# https://docs.opencv.org/4.8.0/d4/db2/classcv_1_1aruco_1_1Board.html
# https://docs.opencv.org/4.8.0/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html
#

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

########
# From
# https://docs.opencv.org/4.8.0/d0/d3c/classcv_1_1aruco_1_1CharucoBoard.html
#
# Parameters
# size	        number of chessboard squares in x and y directions
# squareLength	squareLength chessboard square side length (normally in meters)
# markerLength	marker side length (same unit than squareLength)
# dictionary	dictionary of markers indicating the type of markers
# ids	        array of id used markers The first markers in the dictionary are used to fill the white chessboard squares.
########

board = aruco.CharucoBoard(size=(5, 7), squareLength=0.04, markerLength=0.02, dictionary=aruco_dict)
