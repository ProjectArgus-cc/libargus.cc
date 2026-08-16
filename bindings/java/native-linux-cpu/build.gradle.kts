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

sourceSets {
    main {
        resources {
            srcDir(layout.buildDirectory.dir("generated/resources"))
        }
    }
}
