plugins {
    `java-library`
}

group = "cc.projectargus"
val versionFile = file("${project.rootDir}/version.txt")
version = versionFile.readText().trim()

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

dependencies {
    implementation(project(":bindings-java-core"))
}

val copyNativeLibrary = tasks.register<Copy>("copyNativeLibrary") {
    val osName = System.getProperty("os.name").lowercase()
    val osArch = System.getProperty("os.arch").lowercase()
    
    val osDir = when {
        osName.contains("linux") -> "linux-$osArch"
        osName.contains("windows") -> "windows-$osArch"
        osName.contains("mac") -> "macos-$osArch"
        else -> "unknown"
    }
    
    val libExtension = when {
        osName.contains("windows") -> "argus.dll"
        osName.contains("mac") -> "libargus.dylib"
        else -> "libargus.so"
    }
    
    val buildDir = file("${project.rootDir}/build")
    from(buildDir) {
        include(libExtension)
    }
    into(layout.buildDirectory.dir("generated/resources/natives/$osDir/rocm"))

    onlyIf {
        file("${buildDir}/${libExtension}").exists()
    }
}

sourceSets {
    main {
        resources {
            srcDir(layout.buildDirectory.dir("generated/resources"))
        }
    }
}

tasks.processResources {
    dependsOn(copyNativeLibrary)
}
