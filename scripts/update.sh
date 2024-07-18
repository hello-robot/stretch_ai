echo "=============================================="
echo "         UPDATING STRETCH AI DEPENDENCIES"
echo "=============================================="
echo "---------------------------------------------"
echo "This will update Stretch AI requirements, especially stretch body."
echo "Run this script on your robot, not on the development machine."
echo "Fleet ID: $HELLO_FLEET_ID"
echo "Fleet path: $HELLO_FLEET_PATH"
echo "Ensure stretch_urdf is up-to-date before running this script."
echo "---------------------------------------------"
read -p "Does all this look correct? (y/n) " yn
case $yn in
    y ) echo "Starting installation..." ;;
    n ) echo "Exiting...";
        exit ;;
    * ) echo Invalid response!;
        exit 1 ;;
esac
set -e
pip install --upgrade hello-robot-stretch-body hello-robot-stretch-urdf
pip install numpy<2.0.0 --upgrade
pip install trimesh --upgrade

echo "---------------------------------------------"
echo "Running stretch_urdf commands."
stretch_urdf_ros_update.py
stretch_urdf_ros_update.py --ros2_rebuild

echo "=============================================="
echo "         UPDATE COMPLETE"
echo "=============================================="
