rootProject.name = "libargus"

include("bindings-java-core")
project(":bindings-java-core").projectDir = file("bindings/java/core")

include("bindings-java-platform-cpu")
project(":bindings-java-platform-cpu").projectDir = file("bindings/java/platform-cpu")

include("bindings-java-platform-cuda")
project(":bindings-java-platform-cuda").projectDir = file("bindings/java/platform-cuda")

include("bindings-java-platform-rocm")
project(":bindings-java-platform-rocm").projectDir = file("bindings/java/platform-rocm")

include("bindings-java-platform-vulkan")
project(":bindings-java-platform-vulkan").projectDir = file("bindings/java/platform-vulkan")

