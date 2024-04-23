import setuptools

__version__ = None
with open("stretch/versions.py") as f:
    exec(f.read())  # overrides __version__

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stretchpy",
    version=__version__,
    author="Hello Robot Inc.",
    author_email="support@hello-robot.com",
    description="Stretch Python API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hello-robot/stretchpy",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch" "pyyaml",
        "pyzmq",
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
        "trimesh",
        "openai-clip",
        "yacs",
        "loguru",
        "atomicwrites",
        "transformers",
        "scikit-image",
        "scikit-fmm",
        "pybullet",
        "sophuspy",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "flake8",
        ]
    },
)
