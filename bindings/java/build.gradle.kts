plugins {
    `java-library`
}

group = "cc.projectargus"
version = "0.2.3"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

sourceSets {
    main {
        resources {
            srcDir(layout.buildDirectory.dir("generated/resources"))
        }
    }
}

repositories {
    mavenCentral()
}

dependencies {
    // JUnit 5 for verification and integration testing
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

val copyNativeLibrary = tasks.register<Copy>("copyNativeLibrary") {
    dependsOn(":compileCMake")
    
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
    
    from("${project.rootDir}/build") {
        include(libExtension)
    }
    into(layout.buildDirectory.dir("generated/resources/natives/$osDir"))
}

tasks.processResources {
    dependsOn(copyNativeLibrary)
}

tasks.test {
    useJUnitPlatform()
    
    // Enable native access warning suppression for FFM downcalls in test suite
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
