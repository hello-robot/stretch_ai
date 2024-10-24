# Robocasa Installation

You can install Robocasa by following the instructions below, or you can try the [install script](scripts/install_robocasa.sh).

## Install Robosuite

```bash
git clone https://github.com/ARISE-Initiative/robosuite -b robocasa_v0.1
cd robosuite
pip install -e .
```

## Install Robocasa

```bash
cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
```

## Install Stretch Mujoco

```bash
cd ..
git clone git@github.com:hello-robot/stretch_mujoco.git
cd stretch_mujoco
pip install -e .
```

## Download assets

```bash
python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
python robocasa/scripts/setup_macros.py              # Set up system variables.
```

## Run Robocasa

In one terminal start the server:

```bash
python -m stretch.simulation.mujoco_server --use-robocasa
```

In another run an app, like mapping:

```bash
```
