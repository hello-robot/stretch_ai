# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import cv2
import sys
sys.path.append('/home/hello-robot/repos/dino-vit-features')
from correspondences import find_correspondences, visualize_correspondences
from extractor import ViTExtractor
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from stretch.agent import RobotClient
from stretch.perception.detection.detic import DeticPerception

# image_path1 = "/home/hello-robot/ee_rgb_1.png" #@param
# image_path2 = "/home/hello-robot/ee_rgb_0.png" #@param
# image1 = cv2.imread(image_path1)
# image2 = cv2.imread(image_path2)
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

#@markdown Choose number of points to output:
num_pairs = 10 #@param
#@markdown Choose loading size:
load_size = 224 #@param
#@markdown Choose layer of descriptor:
layer = 9 #@param
#@markdown Choose facet of descriptor:
facet = 'key' #@param
#@markdown Choose if to use a binned descriptor:
bin=True #@param
#@markdown Choose fg / bg threshold:
thresh=0.05 #@param
#@markdown Choose model type:
model_type='dino_vits8' #@param
#@markdown Choose stride:
stride=4 #@param


track_object_id = 41  # detic object id for cup


class Dinobot:
    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4):
        self.robot = RobotClient(robot_ip="10.0.0.14")
        self.bottleneck_image = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = ViTExtractor(model_type = model_type, 
                                              stride = stride, 
                                              device = self.device)
    
    def get_correspondences(self, image1, image2,
                            num_pairs=10, load_size=224, layer=9,
                            facet='key', bin=True, thresh=0.05):
        points1, points2, image1_pil, image2_pil = find_correspondences(self.feature_extractor,
                                                                        image1, image2, 
                                                                        num_pairs, 
                                                                        load_size, 
                                                                        layer, 
                                                                        facet, 
                                                                        bin, 
                                                                        thresh)
        return points1, points2

    def run(self, visualize=False):
        print("Running Dinobot")
        while True:
            obs = self.robot.get_observation()
            servo = self.robot.get_servo_observation()
            ee_rgb = cv2.cvtColor(servo.ee_rgb, cv2.COLOR_BGR2RGB)
            ee_depth = servo.ee_depth
            if not isinstance(self.bottleneck_image,type(None)):
                start = time.perf_counter()
                with torch.no_grad():
                    points1, points2 = self.get_correspondences(self.bottleneck_image, servo.ee_rgb)
                inf_ts = (time.perf_counter() - start) * 1000
                print(f"\n  current: {inf_ts} ms")
                if visualize:
                    if len(points1) == len(points2):
                        im1, im2 = visualize_correspondences(points1, points2, self.bottleneck_image, servo.ee_rgb)
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(im1)
                        axes[0].set_title('Bottleneck Image')
                        axes[0].axis('off')
                        axes[1].imshow(im2)
                        axes[1].set_title('Live Image')
                        axes[1].axis('off')
                        plt.show()
                    else:
                        print("No correspondences found")
    
    def update_bottleneck_image(self, image):
        self.bottleneck_image = image
                

if __name__ == "__main__":
    dinobot = Dinobot()
    detic = DeticPerception()

    bottleneck_image = dinobot.robot.get_servo_observation().ee_rgb
    semantic, instance, task_observations = detic.predict(bottleneck_image)
    if track_object_id in task_observations["instance_classes"]:
        object_mask = semantic == track_object_id
        bottleneck_image[~object_mask] = [0, 0, 0]

    dinobot.update_bottleneck_image(bottleneck_image)
    dinobot.run(visualize=True)