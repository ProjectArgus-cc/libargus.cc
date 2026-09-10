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
    ca-certificates \
    pkg-config \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    glslc \
    libshaderc-dev \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install an immutable, checksum-verified JDK 22 for Panama FFM compatibility.
ARG TEMURIN_22_SHA256=05cd9359dacb1a1730f7c54f57e0fed47942a5292eb56a3a0ee6b13b87457a43
RUN curl --fail --location --silent --show-error \
        https://github.com/adoptium/temurin22-binaries/releases/download/jdk-22.0.2%2B9/OpenJDK22U-jdk_x64_linux_hotspot_22.0.2_9.tar.gz \
        -o /tmp/jdk22.tar.gz \
    && echo "${TEMURIN_22_SHA256}  /tmp/jdk22.tar.gz" | sha256sum --check --strict \
    && mkdir -p /usr/local/jdk-22 \
    && tar -xzf /tmp/jdk22.tar.gz -C /usr/local/jdk-22 --strip-components=1 \
    && rm /tmp/jdk22.tar.gz

# Configure global ccache directory
ENV CCACHE_DIR=/root/.cache/ccache
RUN mkdir -p ${CCACHE_DIR}

WORKDIR /workspace

CMD ["/bin/bash"]
