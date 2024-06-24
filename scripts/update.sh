echo "=============================================="
echo "         UPDATING STRETCH AI DEPENDENCIES"
echo "=============================================="
echo "---------------------------------------------"
echo "This will update Stretch AI requirements, especially stretch body."
set -e
pip install --upgrade hello-robot-stretch-body hello-robot-stretch-urdf
