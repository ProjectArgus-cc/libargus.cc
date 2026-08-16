rootProject.name = "libargus"

include("bindings-java-core")
project(":bindings-java-core").projectDir = file("bindings/java/core")

include("libargus-native-linux-cpu")
project(":libargus-native-linux-cpu").projectDir = file("bindings/java/native-linux-cpu")

include("libargus-native-windows-cpu")
project(":libargus-native-windows-cpu").projectDir = file("bindings/java/native-windows-cpu")

include("libargus-native-linux-cuda")
project(":libargus-native-linux-cuda").projectDir = file("bindings/java/native-linux-cuda")

include("libargus-native-windows-cuda")
project(":libargus-native-windows-cuda").projectDir = file("bindings/java/native-windows-cuda")

include("libargus-native-linux-rocm")
project(":libargus-native-linux-rocm").projectDir = file("bindings/java/native-linux-rocm")

include("libargus-native-linux-vulkan")
project(":libargus-native-linux-vulkan").projectDir = file("bindings/java/native-linux-vulkan")

include("libargus-native-windows-vulkan")
project(":libargus-native-windows-vulkan").projectDir = file("bindings/java/native-windows-vulkan")

include("libargus-native-macos-metal")
project(":libargus-native-macos-metal").projectDir = file("bindings/java/native-macos-metal")


