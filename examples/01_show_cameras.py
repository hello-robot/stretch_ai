import matplotlib

import stretch

# Enable a GUI backend for matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# Connect to the robot
stretch.connect()


nav_cam_image = stretch.take_nav_picture()
plt.imshow(nav_cam_image)
plt.title("Head Nav Cam Image")
plt.axis("off")
plt.show()

ee_msg = stretch.get_ee_frame()
head_msg = stretch.get_head_frame()

ee_color_img = ee_msg["color_image"]
head_color_img = head_msg["color_image"]
ee_depth_img = ee_msg["depth_image"]
head_depth_img = head_msg["depth_image"]

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(ee_color_img)
axs[0, 0].set_title("End Effector Color Image")
axs[0, 1].imshow(ee_depth_img)
axs[0, 1].set_title("End Effector Depth Image")
axs[1, 0].imshow(head_color_img)
axs[1, 0].set_title("Head Color Image")
axs[1, 1].imshow(head_depth_img)
axs[1, 1].set_title("Head Depth Image")
plt.show()
