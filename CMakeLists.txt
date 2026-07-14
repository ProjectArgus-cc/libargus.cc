cmake_minimum_required(VERSION 3.24)
project(libargus VERSION 0.2.0 LANGUAGES C CXX)

# Force modern C++ standards across all compiled objects
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Enforce Position Independent Code for Project Panama shared-library mapping
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# =========================================================================
# 1. Global Hardware Acceleration Pipeline Configuration
# =========================================================================
option(GGML_CUDA "Compile with NVIDIA CUDA Core acceleration" ON)
option(GGML_METAL "Compile with Apple Silicon Metal acceleration" OFF)

if(GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
    # Enable explicit compiler optimizations for target GPU architectures
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()

# =========================================================================
# 2. Dependency Isolation & Upstream Bloat Suppression
# =========================================================================

# Force static compilation of upstream components so they merge directly into libargus
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "Force static compilation of upstream components so they merge directly into libargus")

# Disable all non-essential upstream components before loading build targets
set(LLAMA_BUILD_EXAMPLES    OFF CACHE INTERNAL "Disable llama.cpp CLI applications")
set(LLAMA_BUILD_SERVER      OFF CACHE INTERNAL "Disable llama.cpp embedded server binary")
set(LLAMA_BUILD_TESTS       OFF CACHE INTERNAL "Disable llama.cpp testing footprint")
set(LLAMA_BUILD_MTMD        ON CACHE INTERNAL "Force building the mtmd library standalone")

set(WHISPER_BUILD_EXAMPLES  OFF CACHE INTERNAL "Disable whisper.cpp CLI tools")
set(WHISPER_BUILD_TESTS     OFF CACHE INTERNAL "Disable whisper.cpp validation tests")

# Fetch core llama.cpp framework layer
FetchContent_Declare(
        llamacpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE  # Enforces automatic depth-1 shallow checkout passes
)

# Fetch core whisper.cpp framework layer
FetchContent_Declare(
        whispercpp
        GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
)

# Make components available sequentially; whisper automatically inherits the
# pre-existing initialized 'ggml' targets exposed by the llama subproject
FetchContent_MakeAvailable(llamacpp whispercpp)

# =========================================================================
# 3. Target Assembly Configuration (libargus Public Layer)
# =========================================================================

# Gather all local internal source files
set(ARGUS_SOURCES
        src/argus_common.cc
        src/argus_text.cc
        src/argus_audio.cc
        src/argus_multimodal.cc
)

# Create the final single process-global shared library target
add_library(argus SHARED ${ARGUS_SOURCES})

# Explicitly map unmanaged include paths for compiler visibility
target_include_directories(argus PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${llamacpp_SOURCE_DIR}/ggml/include>
        $<BUILD_INTERFACE:${llamacpp_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${llamacpp_SOURCE_DIR}/tools/mtmd>
        $<BUILD_INTERFACE:${whispercpp_SOURCE_DIR}/include>
)

# Statically link libargus directly into the isolated tensor primitives
target_link_libraries(argus PRIVATE
        llama
        whisper
        ggml
        mtmd
)

# Enforce clean hardware optimization flags across target compilation matrices
if(MSVC)
    target_compile_options(argus PRIVATE /W4 /O2 /Oarch:AVX2)
else()
    target_compile_options(argus PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Enable integrated unit tests
enable_testing()

add_executable(test_libargus tests/test_libargus.cc)
target_include_directories(test_libargus PRIVATE include)
target_link_libraries(test_libargus PRIVATE argus)

add_test(NAME test_libargus COMMAND test_libargus)