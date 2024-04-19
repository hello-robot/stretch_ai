# StretchPy

Requires Python 3.9 or above. **Development Notice**: The code in this repo is a work-in-progress. The code in this repo may be unstable, since we are actively conducting development. Since we have performed limited testing, you may encounter unexpected behaviors.

## Quickstart

This package is not yet available on PyPi. Clone this repo on your Stretch and PC, and install it locally using pip:

```
cd stretchpy/src
pip3 install .
```

On your Stretch, start the server:

```
python3 -m stretch.serve
```

On your PC, add the following yaml to `~/.stretch/config.yaml`:

```yaml
robots:
  - ip_addr: 192.168.1.14 # Substitute with your robot's ip address
    port: 20200
```

Then, on your PC, write some code:

```python
import stretch
stretch.connect()

stretch.move_by(joint_arm=0.1)

for img in stretch.stream_nav_camera():
    cv2.imshow('Nav Camera', img)
    cv2.waitKey(1)
```

## Development

### Pre Commit Hooks

```
pip install pre-commit
pre-commit install
```
