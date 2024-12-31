# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pkg_resources

models_path = pkg_resources.resource_filename("stretch", "simulation/models")
default_scene_xml_path = models_path + "/scene.xml"


def get_default_scene_path() -> str:
    """Return the default scene.xml path."""
    return default_scene_xml_path
