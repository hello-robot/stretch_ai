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
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

COPY install.sh .
RUN chmod +x install.sh
RUN ./install.sh

# Copy requirements file (if you have one)
# COPY requirements.txt .

# Install Python packages (uncomment and modify as needed)
# RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint
CMD ["/bin/bash"]

