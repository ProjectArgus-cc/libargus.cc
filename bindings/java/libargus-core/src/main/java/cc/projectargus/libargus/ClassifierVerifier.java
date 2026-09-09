package cc.projectargus.libargus;

import cc.projectargus.libargus.internal.ArgusBindings;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/** Isolated release verifier; build/load evidence is independent of GPU execution. */
public final class ClassifierVerifier {
    private ClassifierVerifier() {}
    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2) throw new IllegalArgumentException("Expected target [tiny-model]");
        Map<String, String> result = new LinkedHashMap<>(ArgusBackend.getBuildInfo());
        long actual = ArgusBackend.getBuildFeatures();
        long expected = ArgusBackend.expectedFeatureMask(args[0]);
        var packaged = new java.util.Properties();
        try (InputStream metadata = ClassifierVerifier.class.getResourceAsStream("/argus-build.properties")) {
            if (metadata == null) throw new IllegalStateException("Core build identity is absent");
            packaged.load(metadata);
        }
        for (String key : new String[]{"version", "abi_header_sha256"})
            if (!packaged.getProperty(key, "").equals(result.get(key)))
                throw new IllegalStateException("Core/native build identity differs: " + key);
        if (!"local".equals(packaged.getProperty("source")) && !packaged.getProperty("source", "").equals(result.get("source")))
            throw new IllegalStateException("Core/native source revisions differ");
        result.put("expected_features", Long.toString(expected));
        String path = ArgusBindings.loadedLibraryPath();
        if (path == null || ArgusBindings.loadedProvider() == null)
            throw new IllegalStateException("SPI origin was not recorded");
        MessageDigest sha = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(Path.of(path))) {
            byte[] buffer = new byte[1024 * 1024];
            for (int n; (n = in.read(buffer)) >= 0;) sha.update(buffer, 0, n);
        }
        result.put("native_sha256", HexFormat.of().formatHex(sha.digest()));
        result.put("provider", ArgusBindings.loadedProvider());
        result.put("native_path", path);
        result.put("result", actual == expected ? "ok" : "feature_mismatch");
        if (actual == expected && args.length == 2) {
            ArgusBackend.init();
            try (Arena arena = Arena.ofConfined();
                 ArgusModel model = ArgusModel.load(arena, Path.of(args[1]), 0, false);
                 ArgusContext ctx = ArgusContext.init(model, ArgusContextConfig.createDefault(128))) {
                var tokens = arena.allocateFrom(ValueLayout.JAVA_INT, 1, 4, 5);
                if (ctx.decodeBatch(tokens, 3, 0, 0, false) != 0 || ctx.getSeqPosMax(0) != 2)
                    throw new IllegalStateException("Final classifier failed CPU decode");
                result.put("cpu_decode", "passed");
            } finally { ArgusBackend.free(); }
        }
        System.out.println("ARGUS_RECEIPT " + result.entrySet().stream()
            .map(e -> quote(e.getKey()) + ":" + quote(e.getValue()))
            .collect(Collectors.joining(",", "{", "}")));
        if (actual != expected) System.exit(3);
    }
    private static String quote(String value) {
        StringBuilder out = new StringBuilder("\"");
        for (char c : value.toCharArray()) {
            if (c == '"' || c == '\\') out.append('\\').append(c);
            else if (c < 32) out.append(String.format("\\u%04x", (int)c));
            else out.append(c);
        }
        return out.append('"').toString();
    }
}
