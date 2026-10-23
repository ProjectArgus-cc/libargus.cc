plugins {
    `java-library`
}

base {
    archivesName.set("libargus-core")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(providers.gradleProperty("testJdk").orElse("22").get()))
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(22)
}

evaluationDependsOn(":libargus-common")

dependencies {
    api(project(":libargus-common"))

    // JUnit 5 for verification and integration testing
    testImplementation(project.project(":libargus-common").sourceSets.getByName("test").output)
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
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
        val resolved = if (file(it).exists()) file(it) else rootProject.file(it)
        systemProperty("cc.projectargus.libargus.path", resolved.absolutePath)
    }
    systemProperty("argus.testing", providers.gradleProperty("nativeTesting").orElse("false").get())
    systemProperty("argus.root", rootProject.projectDir.absolutePath)

    // Enable native access warning suppression for FFM downcalls in test suite
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-opens=java.base/java.io=ALL-UNNAMED")
    testLogging {
        events("started", "failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
