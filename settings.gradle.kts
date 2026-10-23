rootProject.name = "libargus"

include("libargus-common")
project(":libargus-common").projectDir = file("bindings/java/libargus-common")

include("libargus-core")
project(":libargus-core").projectDir = file("bindings/java/libargus-core")

include("libargus-jni")
project(":libargus-jni").projectDir = file("bindings/java/libargus-jni")

include("libargus-langchain4j")
project(":libargus-langchain4j").projectDir = file("bindings/java/libargus-langchain4j")

include("libargus-native-linux-cpu")
project(":libargus-native-linux-cpu").projectDir = file("bindings/java/libargus-native-linux-cpu")

include("libargus-native-windows-cpu")
project(":libargus-native-windows-cpu").projectDir = file("bindings/java/libargus-native-windows-cpu")

include("libargus-native-linux-cuda")
project(":libargus-native-linux-cuda").projectDir = file("bindings/java/libargus-native-linux-cuda")

include("libargus-native-windows-cuda")
project(":libargus-native-windows-cuda").projectDir = file("bindings/java/libargus-native-windows-cuda")

include("libargus-native-linux-rocm")
project(":libargus-native-linux-rocm").projectDir = file("bindings/java/libargus-native-linux-rocm")

include("libargus-native-linux-vulkan")
project(":libargus-native-linux-vulkan").projectDir = file("bindings/java/libargus-native-linux-vulkan")

include("libargus-native-windows-vulkan")
project(":libargus-native-windows-vulkan").projectDir = file("bindings/java/libargus-native-windows-vulkan")

include("libargus-native-macos-metal")
project(":libargus-native-macos-metal").projectDir = file("bindings/java/libargus-native-macos-metal")
