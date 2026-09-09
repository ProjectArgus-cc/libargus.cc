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
    
    inputs.properties(mapOf("cuda" to useCuda, "metal" to useMetal, "hip" to useHip,
        "vulkan" to useVulkan, "portable" to true))
    inputs.dir("cmake")
    inputs.file("CMakeLists.txt")
    inputs.file("version.txt")
    inputs.dir("include")
    outputs.file("build/CMakeCache.txt")
    
    commandLine(
        "cmake", "-B", "build", 
        "-DCMAKE_BUILD_TYPE=Release", "-DARGUS_PORTABLE=ON",
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
    inputs.properties(mapOf("cuda" to useCuda, "metal" to useMetal, "hip" to useHip,
        "vulkan" to useVulkan, "portable" to true))
    inputs.dir("cmake")
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

        tasks.withType<AbstractArchiveTask>().configureEach {
            isPreserveFileTimestamps = false
            isReproducibleFileOrder = true
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
                        description.set("Native text, speech and vision execution through the Java FFM API.")
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
                maven {
                    name = "Candidate"
                    url = uri(rootProject.layout.buildDirectory.dir("maven"))
                }
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

tasks.register<Exec>("verifyClassifierRuntime") {
    group = "verification"
    description = "Verify an explicit sealed candidate with fresh JVMs and exact receipts"
    doFirst {
        val candidate = providers.gradleProperty("candidateDir").orNull
            ?: error("Supply -PcandidateDir=<sealed candidate> and -Ptarget=<catalog id>")
        val target = providers.gradleProperty("target").orNull
            ?: error("Supply -Ptarget=<catalog id>, for example linux-amd64-cpu")
        val args = mutableListOf("python3", "scripts/release/verify_classifier.py",
            file(candidate).absolutePath, target, layout.buildDirectory.file("receipts/$target.json").get().asFile.absolutePath)
        if (providers.gradleProperty("developmentCandidate").orNull == "true") args.add("--development")
        commandLine(args)
    }
}
