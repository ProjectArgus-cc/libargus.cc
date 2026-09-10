import java.security.MessageDigest

plugins {
    `java-library`
}

base {
    archivesName.set("libargus-core")
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
    options.release.set(22)
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
        val fingerprint = MessageDigest.getInstance("SHA-256").digest(header.readBytes())
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

val skipCMake = project.findProperty("skipCMake")?.toString()?.lowercase().let { it == "true" || it == "on" || it == "1" }

tasks.test {
    if (!skipCMake) {
        dependsOn(":compileCMake")
    }
    useJUnitPlatform()
    val testJdk = providers.gradleProperty("testJdk").orElse("22")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(testJdk.get())) })
    providers.gradleProperty("nativePath").orNull?.let {
        systemProperty("cc.projectargus.libargus.path", file(it).absolutePath)
    }
    systemProperty("argus.testing", providers.gradleProperty("nativeTesting").orElse("false").get())
    systemProperty("argus.root", rootProject.projectDir.absolutePath)
    
    // Enable native access warning suppression for FFM downcalls in test suite
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    testLogging {
        events("started", "failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
