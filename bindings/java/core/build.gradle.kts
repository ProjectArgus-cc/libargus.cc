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

val copyVersionFile = tasks.register<Copy>("copyVersionFile") {
    from("${project.rootDir}/version.txt")
    into(layout.buildDirectory.dir("generated/resources"))
}

tasks.processResources {
    dependsOn(copyVersionFile)
}

tasks.test {
    dependsOn(":compileCMake")
    useJUnitPlatform()
    
    // Enable native access warning suppression for FFM downcalls in test suite
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
