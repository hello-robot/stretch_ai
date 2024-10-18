FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set up timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Python3 dev
RUN apt-get update && apt-get install -y \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install opencv dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6\
    && rm -rf /var/lib/apt/lists/*

# Audio dependencies - for PyAudio
RUN apt-get update && apt-get install -y \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Install ping for network testing
RUN apt-get update && apt-get install -y \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Add espeak for text-to-speech
RUN apt-get update && apt-get install -y \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install mamba
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
# Add mamba to the path
ENV PATH /root/miniforge3/bin:$PATH

# Run our installation script
COPY . .
RUN chmod +x install.sh
RUN ./install.sh -y --no-version

# Configure mamba to start in the correct environment
RUN mamba init

# Add to bashrc so that it starts into the correct environment
RUN echo "mamba activate stretch_ai" >> ~/.bashrc

# ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "stretch_ai", "python", "your_script.py"]

# Copy requirements file (if you have one)
# COPY requirements.txt .

# Install Python packages (uncomment and modify as needed)
# RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint
CMD ["/bin/bash"]

