plugins {
    `java-library`
}

group = "cc.projectargus"
val versionFile = file("${project.rootDir}/version.txt")
version = versionFile.readText().trim()

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(22))
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(17)
}

dependencies {
    implementation(project(":libargus-common"))
}

sourceSets {
    main {
        resources {
            srcDir(layout.buildDirectory.dir("generated/resources"))
        }
    }
}
