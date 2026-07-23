plugins {
    base
}

// Build-time controls for hardware acceleration (e.g. CUDA, Metal)
val useCuda = project.findProperty("cuda")?.toString()?.lowercase().let { it == "true" || it == "on" }
val useMetal = project.findProperty("metal")?.toString()?.lowercase().let { it == "true" || it == "on" }

val libName = when {
    System.getProperty("os.name").lowercase().contains("windows") -> "argus.dll"
    System.getProperty("os.name").lowercase().contains("mac") -> "libargus.dylib"
    else -> "libargus.so"
}

tasks.register<Exec>("configureCMake") {
    group = "build"
    description = "Configures the CMake build directory"
    
    inputs.file("CMakeLists.txt")
    inputs.file("version.txt")
    outputs.file("build/CMakeCache.txt")
    
    commandLine(
        "cmake", "-B", "build", 
        "-DCMAKE_BUILD_TYPE=Release", 
        "-DGGML_CUDA=${if (useCuda) "ON" else "OFF"}",
        "-DGGML_METAL=${if (useMetal) "ON" else "OFF"}"
    )
}

tasks.register<Exec>("compileCMake") {
    group = "build"
    description = "Compiles the native C++ shared library"
    dependsOn("configureCMake")
    
    inputs.dir("src")
    inputs.dir("include")
    inputs.file("CMakeLists.txt")
    inputs.file("version.txt")
    outputs.file("build/lib/$libName")
    
    val nproc = try {
        Runtime.getRuntime().availableProcessors()
    } catch (e: Exception) {
        4
    }
    
    commandLine("cmake", "--build", "build", "--config", "Release", "-j", nproc.toString())
}

tasks.register<Delete>("cleanCMake") {
    group = "build"
    description = "Cleans the CMake build artifacts"
    delete("build")
}

tasks.clean {
    dependsOn("cleanCMake")
}
