# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import glob
import pprint as pp
import time
from datetime import datetime

import cv2
import cv2.aruco as aruco
import webcam as wc
import webcam_calibration_aruco_board as ab
import yaml


def main():
    """Process images and perform webcam calibration."""
    aruco_board = ab.board

    camera_name = "Logitech Webcam C930e"
    image_width = 1920
    image_height = 1080

    image_directory = wc.get_calibration_directory(camera_name, image_width, image_height)
    image_base_name = "webcam_calibration_image"

    file_name_pattern = image_directory + image_base_name + "_*.png"
    file_names = glob.glob(file_name_pattern)

    print("found " + str(len(file_names)) + " calibration images")

    all_object_points = []
    all_image_points = []

    image_size = None
    number_of_images = 0
    number_of_points = 0

    images_used_for_calibration = []

    # Create default parameters
    detector_parameters = aruco.DetectorParameters()
    refine_parameters = aruco.RefineParameters()

    # NOTE: unused older version of aruco detector!
    # aruco_dict = ab.aruco_dict
    # aruco_detector = aruco.ArucoDetector(
    #     aruco_dict, detector_parameters, refine_parameters
    # )

    charuco_parameters = aruco.CharucoParameters()
    charuco_detector = aruco.CharucoDetector(
        aruco_board, charuco_parameters, detector_parameters, refine_parameters
    )

    for f in file_names:
        color_image = cv2.imread(f)
        number_of_images = number_of_images + 1
        if image_size is None:
            image_size = color_image.shape
            print("image_size =", image_size)
        elif image_size != color_image.shape:
            print(
                "ERROR: previous image_size",
                image_size,
                " is not equal to the current image size",
                color_image.shape,
            )
            exit()

        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = charuco_detector.detectBoard(color_image)

        print("filename =", f)

        if (marker_ids is None) or (charuco_ids is None):
            print("marker_ids =", marker_ids)
            print("charuco_ids =", charuco_ids)
        else:
            print("len(charuco_ids) =", len(charuco_ids))

            if len(charuco_ids) > 0:
                object_points, image_points = aruco_board.matchImagePoints(
                    charuco_corners, charuco_ids
                )

                if True:
                    print()
                    print("len(object_points) =", len(object_points))
                    print("len(image_points) =", len(image_points))

                # A view with fewer than eight points results in
                # cv2.calibrateCamera throwing an error like the following:
                # projection_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera( cv2.error: OpenCV(4.8.1) /io/opencv/modules/calib3d/src/calibration.cpp:1213: error: (-215:Assertion failed) fabs(sc) > DBL_EPSILON in function 'cvFindExtrinsicCameraParams2'

                if len(object_points) >= 8:
                    number_of_points = number_of_points + len(object_points)
                    all_object_points.append(object_points)
                    all_image_points.append(image_points)
                    images_used_for_calibration.append(f)

                if False:
                    print()
                    print("********************************")
                    print(" MATCH IMAGE POINTS OUTPUT ")
                    print()
                    print("object_points =", object_points)
                    print("_____")
                    print("image_points =", image_points)
                    print()
                    print("********************************")
                    print()

                aruco.drawDetectedCornersCharuco(color_image, charuco_corners, charuco_ids)

        cv2.imshow("Detected Charuco Corners", color_image)
        cv2.waitKey(1)

    # Perform Calibration
    size = (image_size[1], image_size[0])

    print()
    print("POINTS USED FOR CALIBRATION")
    print("number of images with suitable points =", len(images_used_for_calibration))
    print("len(all_object_points) =", len(all_object_points))
    print("len(all_image_points) =", len(all_image_points))

    (
        projection_error,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    ) = cv2.calibrateCamera(all_object_points, all_image_points, size, None, None)

    calibration_results = {
        "camera_name": camera_name,
        "calibration_date": datetime.now(),
        "image_size": list(image_size),
        "number_of_images_processed": number_of_images,
        "number_of_images_used": len(images_used_for_calibration),
        "number_of_corresponding_points_used": number_of_points,
        "projection_error": projection_error,
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion_coefficients,
    }

    print()
    print("calibration_results =")
    pp.pprint(calibration_results)
    print()

    # Convert from Numpy arrays to human-readable lists
    calibration_results = {
        (k): (v.tolist() if "tolist" in dir(v) else v) for k, v in calibration_results.items()
    }

    results_file_time = time.strftime("%Y%m%d%H%M%S")
    results_file_name = (
        image_directory + "camera_calibration_results_" + results_file_time + ".yaml"
    )
    with open(results_file_name, "w") as file:
        yaml.dump(calibration_results, file, sort_keys=True)
    print("saved calibration results to", results_file_name)
    print()


if __name__ == "__main__":
    main()
