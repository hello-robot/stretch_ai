# Aruco Markers

You can detect ArUco markers using StretchPy. ArUco markers are like QR codes; they're printable visual patterns that can be understood by computers. In addition to encoding data (a single number called the "marker ID"), the 3D pose of an ArUco marker can be estimated by the camera. These qualities make ArUco markers popular for use with robots.

![Mobile base left corner Aruco marker](./images/mobile_base_left_corner_aruco.png)

Stretch uses ArUco markers extensively. There's stickers placed on the robot's body, a single large sticker on Stretch's docking station, and spare markers for you to use in your applications. In this tutorial, we'll cover how to detect and work with ArUco markers.

## Known markers

The robot stores all of the markers it knows about in its marker database. You can query it using:

```python
from stretch.perception.aruco.markers import MarkersDatabase

db = MarkerDatabase()
print(db.get_markers())

# example output:
# {130: {'length_mm': 47.0,
#  'link': 'link_aruco_left_base',
#  'name': 'base_left',
#  'use_rgb_only': False},
# 131: {'length_mm': 47.0,
#  'link': 'link_aruco_right_base',
#  'name': 'base_right',
#  'use_rgb_only': False},
# ...
```

Estimating the marker's pose requires the marker ID and marker size (in millimeters), which is why each marker dictionary consists of:

- The key is the marker ID
- *length_mm* is the length of the marker in millimeters
- *link* is the name of the URDF link that associates with this marker, if any
- *name* is the name of the marker (e.g. 'docking station')
- *use_rgb_only* is a flag that determines whether pose estimation of the marker considers depth data
