# Docker

This is a guide to setting up a Docker container for running the Stretch software. This is useful for running the software on a computer that doesn't have the correct dependencies installed, or for running the software in a controlled environment.

## What is Docker and Why Would I Use It?

Docker is a tool that allows you to run software in a container. A container is a lightweight, standalone, executable package of software that includes everything needed to run the software, including the code, runtime, system tools, libraries, and settings. Containers are isolated from each other and from the host system, so they are a great way to run software in a controlled environment.

In particular, this docker image is designed to use the correct CUDA version for the Stretch software. This is useful if you are running the software on a computer that doesn't have the correct CUDA version installed, and it makes installing the right versions of AI/ML libraries much easier.

## Installing Docker

To install Docker, follow the instructions on the [Docker website](https://docs.docker.com/get-docker/). Test by running:

```
docker run hello-world
```

### Troubleshooting

If you are having trouble installing Docker, check the [Docker documentation](https://docs.docker.com/get-docker/) for troubleshooting tips. If you are getting a permission error on Ubuntu, make sure you have added your user to the `docker` group:

```
# Make sure the group exists
sudo groupadd docker

# Add $USER to the docker group
sudo usermod -aG docker $USER

# Restart the Docker daemon so changes take effect
sudo systemctl restart docker
```

If necessary, you can apply new group changes to current session:

```
newgrp docker
```

## Building the Docker Image

To build the Docker image, clone this repository and run the following command in the root directory of the repository:

```bash
docker build -t stretch-ai_cuda-11.8:latest .
```

### Placeholder: Building and Pushing to Dockerhub

This will use the Hello Robot account as an example (username: `hellorobotinc`). Login with:

```
docker login -u hellorobotinc
```

and enter a password (or create an [access token](https://hub.docker.com/settings/security)).

Then, build the image with:

```bash
docker build -t hellorobotinc/stretch-ai_cuda-11.8:latest .
docker push hellorobotinc/stretch-ai_cuda-11.8:latest
```

You can pull with:

```bash
docker pull hellorobotinc/stretch-ai_cuda-11.8:latest
```
