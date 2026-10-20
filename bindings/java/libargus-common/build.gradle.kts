import java.security.MessageDigest

plugins {
    `java-library`
}

base {
    archivesName.set("libargus-common")
}

group = "cc.projectargus"

val versionFile = file("${project.rootDir}/version.txt")
version = versionFile.readText().trim()

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(providers.gradleProperty("testJdk").orElse("22").get()))
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(17)
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
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

val copyVersionFile = tasks.register<Copy>("copyVersionFile") {
    from("${project.rootDir}/version.txt")
    into(layout.buildDirectory.dir("generated/resources"))
}

val copyBuildInfo = tasks.register("copyBuildInfo") {
    val source = providers.gradleProperty("sourceRevision").orElse("local")
    val header = rootProject.file("include/libargus.h")
    val destination = layout.buildDirectory.file("generated/resources/argus-build.properties")
    inputs.property("source", source)
    inputs.property("version", project.version.toString())
    inputs.file(header)
    outputs.file(destination)
    doLast {
        val revision = source.get()
        require(revision == "local" || revision.matches(Regex("[0-9a-f]{40}"))) { "Invalid sourceRevision" }
        val canonicalHeader = header.readText(Charsets.UTF_8).replace("\r\n", "\n")
        val fingerprint = MessageDigest.getInstance("SHA-256").digest(canonicalHeader.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        val output = destination.get().asFile
        output.parentFile.mkdirs()
        output.writeText("version=${project.version}\nsource=$revision\nabi_header_sha256=$fingerprint\n")
    }
}

tasks.processResources {
    dependsOn(copyVersionFile)
    dependsOn(copyBuildInfo)
}

tasks.named("sourcesJar") {
    dependsOn(copyVersionFile)
    dependsOn(copyBuildInfo)
}

tasks.test {
    useJUnitPlatform()
    val testJdk = providers.gradleProperty("testJdk").orElse("22")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(testJdk.get())) })
    testLogging {
        events("started", "failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
