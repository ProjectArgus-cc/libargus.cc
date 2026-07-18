#!/usr/bin/env bash
# =========================================================================
# package-artifacts.sh — Assemble compiled native binaries into resource trees
# =========================================================================
# Helper script to copy compiled dynamic libraries (.so, .dll, .dylib) 
# from the build-grid artifact distributions into the respective
# modular Java platform project resource paths.
#
# Usage:
#   ./scripts/package-artifacts.sh [dist_dir] [bindings_dir] [library_name]
# =========================================================================

set -euo pipefail

DIST_DIR="${1:-dist}"
BINDINGS_DIR="${2:-bindings/java}"
LIB_NAME="${3:-argus}"

if [[ ! -d "${DIST_DIR}" ]]; then
    echo "Error: Directory '${DIST_DIR}' does not exist. Please run builds first."
    exit 1
fi

if [[ ! -d "${BINDINGS_DIR}" ]]; then
    echo "Error: Directory '${BINDINGS_DIR}' does not exist."
    exit 1
fi

echo "========================================================================="
echo "libargus.cc Artifact Assembler [Lib: ${LIB_NAME}]"
echo "========================================================================="

# Helper copy function
copy_if_exists() {
    local src="$1"
    local dest_dir="$2"
    local dest_file="$3"
    
    if [[ -f "${src}" ]]; then
        echo "Packaging: ${src} -> ${dest_dir}/${dest_file}"
        mkdir -p "${dest_dir}"
        cp "${src}" "${dest_dir}/${dest_file}"
    else
        echo "Skip (Not Found): ${src}"
    fi
}

# --- 1. Linux Targets (AMD64) ---
copy_if_exists \
    "${DIST_DIR}/linux-amd64/cpu/lib${LIB_NAME}.so" \
    "${BINDINGS_DIR}/platform-cpu/src/main/resources/natives/linux-amd64/cpu" \
    "lib${LIB_NAME}.so"

copy_if_exists \
    "${DIST_DIR}/linux-amd64/cuda/lib${LIB_NAME}.so" \
    "${BINDINGS_DIR}/platform-cuda/src/main/resources/natives/linux-amd64/cuda" \
    "lib${LIB_NAME}.so"

copy_if_exists \
    "${DIST_DIR}/linux-amd64/rocm/lib${LIB_NAME}.so" \
    "${BINDINGS_DIR}/platform-rocm/src/main/resources/natives/linux-amd64/rocm" \
    "lib${LIB_NAME}.so"

copy_if_exists \
    "${DIST_DIR}/linux-amd64/vulkan/lib${LIB_NAME}.so" \
    "${BINDINGS_DIR}/platform-vulkan/src/main/resources/natives/linux-amd64/vulkan" \
    "lib${LIB_NAME}.so"

# --- 2. Windows Targets (AMD64) ---
copy_if_exists \
    "${DIST_DIR}/windows-amd64/cpu/${LIB_NAME}.dll" \
    "${BINDINGS_DIR}/platform-cpu/src/main/resources/natives/windows-amd64/cpu" \
    "${LIB_NAME}.dll"

copy_if_exists \
    "${DIST_DIR}/windows-amd64/cuda/${LIB_NAME}.dll" \
    "${BINDINGS_DIR}/platform-cuda/src/main/resources/natives/windows-amd64/cuda" \
    "${LIB_NAME}.dll"

copy_if_exists \
    "${DIST_DIR}/windows-amd64/vulkan/${LIB_NAME}.dll" \
    "${BINDINGS_DIR}/platform-vulkan/src/main/resources/natives/windows-amd64/vulkan" \
    "${LIB_NAME}.dll"

# --- 3. macOS Targets (Universal / Metal) ---
# Map Universal Mac Metal library to both x86_64 and aarch64 CPU paths (automatic runtime loading fallback)
copy_if_exists \
    "${DIST_DIR}/macos-universal/metal/lib${LIB_NAME}.dylib" \
    "${BINDINGS_DIR}/platform-cpu/src/main/resources/natives/macos-x86_64/cpu" \
    "lib${LIB_NAME}.dylib"

copy_if_exists \
    "${DIST_DIR}/macos-universal/metal/lib${LIB_NAME}.dylib" \
    "${BINDINGS_DIR}/platform-cpu/src/main/resources/natives/macos-aarch64/cpu" \
    "lib${LIB_NAME}.dylib"

echo "========================================================================="
echo "Assembly completed successfully. Run gradle tasks to build JARs."
echo "========================================================================="
