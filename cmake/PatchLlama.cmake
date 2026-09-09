# b10472 uses floating literals to initialize HIP 6.1's raw uint16 BF16 lanes.
# Integer zero preserves the same all-zero BF16 representation without narrowing.
set(path "${llamacpp_SOURCE_DIR}/ggml/src/ggml-cuda/mma.cuh")
file(READ "${path}" contents)
set(before "nv_bfloat162 x[ne] = {{0.0f, 0.0f}};")
set(after "nv_bfloat162 x[ne] = {{0, 0}};")
string(FIND "${contents}" "${before}" original)
string(FIND "${contents}" "${after}" patched)
if(NOT original EQUAL -1)
    string(REPLACE "${before}" "${after}" contents "${contents}")
    file(WRITE "${path}" "${contents}")
elseif(patched EQUAL -1)
    message(FATAL_ERROR "Pinned llama.cpp BF16 compatibility patch no longer applies")
endif()
