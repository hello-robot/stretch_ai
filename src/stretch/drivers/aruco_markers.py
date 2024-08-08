# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pathlib

import stretch_body.hello_utils as hu
import yaml
from schema import And, Or, Schema, SchemaError, Use

# The default set of aruco markers used with Stretch is
# defined in this file as a dictionary. For more details
# on Stretch's aruco markers, see docs/arucos.md
DEFAULT_ARUCO_MARKER_INFO = {
    130: {
        "length_mm": 47.0,
        "use_rgb_only": False,
        "name": "base_left",
        "link": "link_aruco_left_base",
    },
    131: {
        "length_mm": 47.0,
        "use_rgb_only": False,
        "name": "base_right",
        "link": "link_aruco_right_base",
    },
    132: {
        "length_mm": 23.5,
        "use_rgb_only": False,
        "name": "wrist_inside",
        "link": "link_aruco_inner_wrist",
    },
    133: {
        "length_mm": 23.5,
        "use_rgb_only": False,
        "name": "wrist_top",
        "link": "link_aruco_top_wrist",
    },
    134: {
        "length_mm": 31.4,
        "use_rgb_only": False,
        "name": "shoulder_top",
        "link": "link_aruco_shoulder",
    },
    135: {
        "length_mm": 31.4,
        "use_rgb_only": False,
        "name": "d405_back",
        "link": "link_aruco_d405",
    },
    200: {
        "length_mm": 14.0,
        "use_rgb_only": True,
        "name": "finger_left",
        "link": "link_finger_left",
    },
    201: {
        "length_mm": 14.0,
        "use_rgb_only": True,
        "name": "finger_right",
        "link": "link_finger_right",
    },
    202: {
        "length_mm": 30.0,
        "use_rgb_only": True,
        "name": "toy",
        "link": "None",
    },
    245: {
        "length_mm": 88.0,
        "use_rgb_only": False,
        "name": "docking_station",
        "link": None,
    },
}


class MarkersDatabase:
    @staticmethod
    def generate_database_from_defaults():
        maps_path = pathlib.Path(hu.get_stretch_directory("maps"))
        if not maps_path.is_dir():
            maps_path.mkdir(parents=True, exist_ok=True)

        marker_info_path = maps_path / "aruco_marker_info.yaml"
        if marker_info_path.is_file():
            marker_info_path.unlink(missing_ok=True)

        with open(str(marker_info_path), "w") as f:
            yaml.dump(DEFAULT_ARUCO_MARKER_INFO, f)

    def __init__(self):
        # load marker info
        self.marker_info_path = (
            pathlib.Path(hu.get_stretch_directory("maps")) / "aruco_marker_info.yaml"
        )
        if not self.marker_info_path.is_file():
            MarkersDatabase.generate_database_from_defaults()
        with open(str(self.marker_info_path), "r") as f:
            self.marker_info = yaml.safe_load(f)

        # set marker schemas
        self.add_schema = Schema(
            {
                And(Use(int), lambda k: 0 <= k <= 99): {
                    "length_mm": Use(float),
                    "use_rgb_only": bool,
                    "name": Or(str, None),
                    "link": Or(str, None),
                }
            }
        )
        self.delete_schema = Schema(And(Use(int), lambda k: 0 <= k <= 99))

    def add(self, new_marker):
        # verify new marker is valid
        try:
            new_marker = self.add_schema.validate(new_marker)
        except SchemaError:
            return "Rejected: new marker doesn't match format"

        # add new marker
        hu.overwrite_dict(self.marker_info, new_marker)

        # save and reload database
        with open(str(self.marker_info_path), "w") as f:
            yaml.dump(self.marker_info, f, sort_keys=True)
        with open(str(self.marker_info_path), "r") as f:
            self.marker_info = yaml.safe_load(f)

        return "Accepted"

    def delete(self, marker_id):
        # verify id to delete is valid
        try:
            marker_id = self.delete_schema.validate(marker_id)
        except SchemaError:
            return f"Rejected: cannot delete marker id {marker_id}"

        # delete marker
        if marker_id in self.marker_info:
            del self.marker_info[marker_id]

            # save and reload database
            with open(str(self.marker_info_path), "w") as f:
                yaml.dump(self.marker_info, f, sort_keys=True)
            with open(str(self.marker_info_path), "r") as f:
                self.marker_info = yaml.safe_load(f)

        return "Accepted"
