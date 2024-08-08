#!/bin/bash

echo "Copy udev rule file for the Logitech Webcam C930e."
echo "sudo cp ./99-hello-dex-teleop-camera.rules /etc/udev/rules.d/"
sudo cp ./99-hello-dex-teleop-camera.rules /etc/udev/rules.d/
echo ""

echo "Activate the new udev rule."
echo "sudo udevadm control --reload"
sudo udevadm control --reload
echo ""

echo "Install v4l2 utilities..."
echo "sudo apt install v4l-utils"
sudo apt install v4l-utils
echo ""

echo "Next, you need to generate specialized URDF files and calibrate your Logitech C930e webcam."
echo "You can find instructions in the README.md file."
echo ""
echo "***********************************************************"
echo "IMPORTANT: After this installation finishes, you should unplug and then replug your Logitech Webcam C930e, if it's already plugged in."
echo "***********************************************************"
echo ""
