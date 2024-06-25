echo "=============================================="
echo "         UPDATING STRETCH AI DEPENDENCIES"
echo "=============================================="
echo "---------------------------------------------"
echo "This will update Stretch AI requirements, especially stretch body."
echo "Run this script on your robot, not on the development machine."
echo "Fleet ID: $HELLO_FLEET_ID"
echo "Fleet path: $HELLO_FLEET_PATH"
echo "---------------------------------------------"
read -p "Does all this look correct? (y/n) " yn
case $yn in
	y ) echo "Starting installation...";;
	n ) echo "Exiting...";
		exit;;
	* ) echo Invalid response!;
		exit 1;;
esac
set -e
pip install --upgrade hello-robot-stretch-body hello-robot-stretch-urdf

echo "---------------------------------------------"
echo "Going to Stretch URDF directory."

echo "=============================================="
echo "         UPDATE COMPLETE"
echo "=============================================="
