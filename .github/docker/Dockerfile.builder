FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/local/jdk-22
ENV PATH=${JAVA_HOME}/bin:${PATH}

# Install core system toolchain, build dependencies, ccache, and Vulkan development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    clang \
    cmake \
    ninja-build \
    ccache \
    git \
    curl \
    wget \
    ca-certificates \
    pkg-config \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    glslc \
    libshaderc-dev \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install OpenJDK 22 for Project Panama (FFM) Java 22+ API compatibility
RUN wget -q https://download.oracle.com/java/22/latest/jdk-22_linux-x64_bin.tar.gz -O /tmp/jdk22.tar.gz \
    && mkdir -p /usr/local/jdk-22 \
    && tar -xzf /tmp/jdk22.tar.gz -C /usr/local/jdk-22 --strip-components=1 \
    && rm /tmp/jdk22.tar.gz

# Configure global ccache directory
ENV CCACHE_DIR=/root/.cache/ccache
RUN mkdir -p ${CCACHE_DIR}

WORKDIR /workspace

CMD ["/bin/bash"]
