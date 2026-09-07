plugins {
    base
    id("com.gradleup.nmcp.aggregation") version "1.6.1"
}

// Build-time controls for hardware acceleration (e.g. CUDA, Metal, ROCm/HIP, Vulkan)
val useCuda = project.findProperty("cuda")?.toString()?.lowercase().let { it == "true" || it == "on" }
val useMetal = project.findProperty("metal")?.toString()?.lowercase().let { it == "true" || it == "on" }
val useHip = (project.findProperty("hip") ?: project.findProperty("rocm"))?.toString()?.lowercase().let { it == "true" || it == "on" }
val useVulkan = project.findProperty("vulkan")?.toString()?.lowercase().let { it == "true" || it == "on" }

val libName = when {
    System.getProperty("os.name").lowercase().contains("windows") -> "argus.dll"
    System.getProperty("os.name").lowercase().contains("mac") -> "libargus.dylib"
    else -> "libargus.so"
}

val skipCMake = project.findProperty("skipCMake")?.toString()?.lowercase().let { it == "true" || it == "on" || it == "1" }

val possibleLibFiles = listOf(
    file("build/lib/$libName"),
    file("build/lib/Release/$libName"),
    file("build/bin/$libName"),
    file("build/bin/Release/$libName"),
    file("build/$libName")
)

tasks.register<Exec>("configureCMake") {
    group = "build"
    description = "Configures the CMake build directory"
    onlyIf { !skipCMake }
    
    inputs.file("CMakeLists.txt")
    inputs.file("version.txt")
    outputs.file("build/CMakeCache.txt")
    
    commandLine(
        "cmake", "-B", "build", 
        "-DCMAKE_BUILD_TYPE=Release", 
        "-DGGML_CUDA=${if (useCuda) "ON" else "OFF"}",
        "-DGGML_METAL=${if (useMetal) "ON" else "OFF"}",
        "-DGGML_HIP=${if (useHip) "ON" else "OFF"}",
        "-DGGML_VULKAN=${if (useVulkan) "ON" else "OFF"}"
    )
}

tasks.register<Exec>("compileCMake") {
    group = "build"
    description = "Compiles the native C++ shared library"
    onlyIf { !skipCMake }
    dependsOn("configureCMake")
    
    inputs.dir("src")
    inputs.dir("include")
    inputs.file("CMakeLists.txt")
    inputs.file("version.txt")
    outputs.files(possibleLibFiles)
    
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

repositories {
    mavenCentral()
}

subprojects {
    repositories {
        mavenCentral()
    }

    plugins.withType<JavaPlugin> {
        apply(plugin = "maven-publish")
        apply(plugin = "signing")
        apply(plugin = "com.gradleup.nmcp")

        configure<JavaPluginExtension> {
            withSourcesJar()
            withJavadocJar()
        }

        tasks.withType<Javadoc> {
            (options as? StandardJavadocDocletOptions)?.apply {
                addStringOption("Xdoclint:none", "-quiet")
                encoding = "UTF-8"
                charSet = "UTF-8"
            }
        }

        configure<PublishingExtension> {
            publications {
                create<MavenPublication>("mavenJava") {
                    from(components["java"])
                    artifactId = project.name

                    pom {
                        name.set(artifactId)
                        description.set("Unmanaged, zero-allocation native AI execution runtime behind Panama FFM boundary.")
                        url.set("https://github.com/ProjectArgus-cc/libargus.cc")

                        licenses {
                            license {
                                name.set("MIT License")
                                url.set("https://opensource.org/licenses/MIT")
                            }
                        }
                        developers {
                            developer {
                                id.set("projectargus")
                                name.set("ProjectArgus Team")
                            }
                        }
                        scm {
                            connection.set("scm:git:git://github.com/ProjectArgus-cc/libargus.cc.git")
                            developerConnection.set("scm:git:ssh://github.com:ProjectArgus-cc/libargus.cc.git")
                            url.set("https://github.com/ProjectArgus-cc/libargus.cc")
                        }
                    }
                }
            }
            repositories {
                // 1. GitHub Packages Maven Registry
                maven {
                    name = "GitHubPackages"
                    url = uri("https://maven.pkg.github.com/ProjectArgus-cc/libargus.cc")
                    credentials {
                        username = System.getenv("GITHUB_ACTOR") ?: project.findProperty("gpr.user")?.toString()
                        password = System.getenv("GITHUB_TOKEN") ?: project.findProperty("gpr.key")?.toString()
                    }
                }
            }
        }

        configure<SigningExtension> {
            val signingKey = System.getenv("GPG_PRIVATE_KEY") ?: project.findProperty("signing.key")?.toString()
            val signingPassphrase = System.getenv("GPG_PASSPHRASE") ?: project.findProperty("signing.password")?.toString()
            if (!signingKey.isNullOrEmpty()) {
                useInMemoryPgpKeys(signingKey, signingPassphrase)
                sign(extensions.getByType<PublishingExtension>().publications["mavenJava"])
            }
        }
    }
}

nmcpAggregation {
    centralPortal {
        username.set(
            providers.gradleProperty("mavenCentralUsername")
                .orElse(providers.environmentVariable("MAVEN_CENTRAL_USERNAME"))
        )
        password.set(
            providers.gradleProperty("mavenCentralPassword")
                .orElse(providers.environmentVariable("MAVEN_CENTRAL_PASSWORD"))
        )
        publishingType.set("AUTOMATIC")
    }
}

dependencies {
    subprojects.forEach { subproject ->
        "nmcpAggregation"(subproject)
    }
}

tasks.register("verifyPackagedClassifiers") {
    group = "verification"
    description = "Verifies that all packaged native classifier JARs contain non-empty binaries under natives/"
    dependsOn(subprojects.map { it.tasks.matching { t -> t.name == "assemble" } })

    doLast {
        val requireAll = (project.findProperty("requireAllNatives") ?: System.getenv("CI"))?.toString()?.lowercase().let { it == "true" || it == "1" }
        val nativeProjects = subprojects.filter { it.name.startsWith("libargus-native-") }
        var verifiedCount = 0
        for (subproject in nativeProjects) {
            val libsDir = subproject.layout.buildDirectory.dir("libs").get().asFile
            val jarFile = libsDir.resolve("${subproject.name}-${subproject.version}.jar")
            if (!jarFile.exists()) {
                if (requireAll) {
                    error("Missing required JAR for native subproject: ${subproject.name} (${jarFile.name})")
                }
                continue
            }
            var foundNative = false
            java.util.zip.ZipFile(jarFile).use { zip ->
                val entries = zip.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    if (entry.name.startsWith("natives/") && !entry.isDirectory) {
                        foundNative = true
                        if (entry.size <= 0) {
                            error("Corrupt native library in ${jarFile.name}: ${entry.name} has size 0")
                        }
                    }
                }
            }
            if (!foundNative) {
                if (requireAll) {
                    error("Packaged classifier JAR ${jarFile.name} contains no native binaries under natives/")
                } else {
                    logger.warn("Packaged classifier JAR ${jarFile.name} has no native binaries (skipping in local dev build without requireAllNatives)")
                }
            } else {
                verifiedCount++
                logger.lifecycle("Verified native payload in ${jarFile.name}")
            }
        }
        if (requireAll && verifiedCount == 0) {
            error("No native classifier JARs could be verified in CI environment!")
        }
        logger.lifecycle("Successfully verified $verifiedCount packaged classifier JAR(s).")
    }
}

tasks.register("verifyClassifierRuntime") {
    group = "verification"
    description = "Loads a target classifier JAR in isolation with libargus-core and verifies runtime extraction and feature mask"

    doLast {
        val targetClassifier = project.findProperty("targetClassifier")?.toString()
            ?: project.findProperty("classifierJar")?.toString()
        val expectedTarget = project.findProperty("expectedTarget")?.toString() ?: ""

        if (targetClassifier.isNullOrEmpty()) {
            logger.lifecycle("No targetClassifier specified; skipping isolated classifier runtime verification.")
            return@doLast
        }

        val coreProj = subprojects.find { it.name == "libargus-core" }
            ?: error("Subproject 'libargus-core' not found!")
        val coreJar = coreProj.layout.buildDirectory.dir("libs").get().asFile.resolve("${coreProj.name}-${coreProj.version}.jar")

        val subJar = if (project.file(targetClassifier).exists()) {
            project.file(targetClassifier)
        } else {
            val subproj = subprojects.find { it.name == targetClassifier }
                ?: error("Subproject or jar '$targetClassifier' not found!")
            subproj.layout.buildDirectory.dir("libs").get().asFile.resolve("${subproj.name}-${subproj.version}.jar")
        }

        if (!subJar.exists()) error("Classifier JAR missing: ${subJar.absolutePath}")
        if (!coreJar.exists()) error("Core JAR missing: ${coreJar.absolutePath}")

        val cp = listOf(subJar.absolutePath, coreJar.absolutePath).joinToString(java.io.File.pathSeparator)
        val javaBin = java.nio.file.Paths.get(System.getProperty("java.home"), "bin", if (System.getProperty("os.name").lowercase().contains("windows")) "java.exe" else "java").toString()

        logger.lifecycle("Executing isolated JVM classifier runtime test with CP: $cp (target=$expectedTarget)")
        val proc = ProcessBuilder(
            javaBin,
            "--enable-native-access=ALL-UNNAMED",
            "-cp", cp,
            "cc.projectargus.libargus.ArgusBackend",
            expectedTarget
        ).redirectErrorStream(true).start()

        val output = proc.inputStream.bufferedReader().readText()
        val exitCode = proc.waitFor()
        logger.lifecycle("Verification output:\n$output")
        if (exitCode != 0) {
            error("Isolated classifier runtime test failed with exit code $exitCode: $output")
        }
        logger.lifecycle("Isolated classifier runtime test passed successfully for $targetClassifier")
    }
}

