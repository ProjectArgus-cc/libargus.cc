plugins {
    `java-library`
}

base {
    archivesName.set("libargus-jni")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(providers.gradleProperty("testJdk").orElse("22").get()))
    }
}

tasks.named<JavaCompile>("compileJava") {
    options.release.set(17)
}

tasks.named<JavaCompile>("compileTestJava") {
    options.release.set(17)
}

dependencies {
    api(project(":libargus-common"))

    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

val skipCMake = project.findProperty("skipCMake")?.toString()?.lowercase().let { it == "true" || it == "on" || it == "1" }

tasks.test {
    if (!skipCMake) {
        dependsOn(":compileCMake")
    }
    dependsOn(":libargus-core:jar")
    useJUnitPlatform()
    val testJdk = providers.gradleProperty("testJdk").orElse("22")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(testJdk.get())) })
    providers.gradleProperty("nativePath").orNull?.let {
        val resolved = if (file(it).exists()) file(it) else rootProject.file(it)
        systemProperty("cc.projectargus.libargus.path", resolved.absolutePath)
    }
    systemProperty("argus.testing", providers.gradleProperty("nativeTesting").orElse("false").get())
    systemProperty("argus.root", rootProject.projectDir.absolutePath)
    systemProperty(
        "argus.core.jar",
        project(":libargus-core").tasks.named<Jar>("jar").get().archiveFile.get().asFile.absolutePath
    )

    // Enable native access for differential tests that load libargus-core (FFM) in an isolated ClassLoader
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-opens=java.base/java.io=ALL-UNNAMED")
    testLogging {
        events("started", "failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
