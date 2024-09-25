from dataclasses import dataclass
from typing import List, Text

from torch import Tensor


@dataclass
class Observations:
    low_level_output_messages: List[str] = None
    scene_images: List = None
    object_images: List = None
    scene_graph: List = None


@dataclass
class ObjectImage:
    image: Tensor = None
    position: List[float] = None
    crop_id: int = None
    object_class: Text = None
    instance_id: int = None
