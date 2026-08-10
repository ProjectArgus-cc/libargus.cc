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

subprojects {
    plugins.withType<JavaPlugin> {
        apply(plugin = "maven-publish")
        apply(plugin = "signing")

        configure<PublishingExtension> {
            publications {
                create<MavenPublication>("mavenJava") {
                    from(components["java"])

                    pom {
                        name.set(project.name)
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

            // 2. Maven Central (Sonatype OSSRH / Central Portal)
            maven {
                name = "MavenCentral"
                url = uri("https://oss.sonatype.org/service/local/staging/deploy/maven2/")
                credentials {
                    username = System.getenv("MAVEN_CENTRAL_USERNAME") ?: project.findProperty("ossrhUsername")?.toString()
                    password = System.getenv("MAVEN_CENTRAL_PASSWORD") ?: project.findProperty("ossrhPassword")?.toString()
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


