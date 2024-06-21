FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set up timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install mamba
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
RUN bash Miniforge3-$(uname)-$(uname -m).sh -b
# Add mamba to the path
ENV PATH /root/miniforge3/bin:$PATH

# Run our installation script
COPY ./src ./src
COPY install.sh .
RUN chmod +x install.sh
RUN ./install.sh -y

# Copy requirements file (if you have one)
# COPY requirements.txt .

# Install Python packages (uncomment and modify as needed)
# RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint
CMD ["/bin/bash"]

