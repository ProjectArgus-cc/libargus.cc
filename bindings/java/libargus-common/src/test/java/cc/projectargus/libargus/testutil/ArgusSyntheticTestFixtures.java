package cc.projectargus.libargus.testutil;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 * Pure-Java deterministic synthetic binary fixture generator for {@code libargus} tests and doctests.
 *
 * <p>Synthesizes tiny, valid CPU-executable model and media fixtures on the fly into JUnit
 * {@code @TempDir} directories with zero repository binary bloat:
 * <ul>
 *   <li>{@link #generateTinyWhisperModel(Path)}: GGML Whisper ASR model ({@code tiny-whisper.bin})</li>
 *   <li>{@link #generateTinyOmniProjector(Path)}: Unified Vision ({@code mlp}) + Audio ({@code gemma4ua})
 *       {@code libmtmd} GGUF v3 projector ({@code tiny-mmproj-omni.gguf})</li>
 *   <li>{@link #generateTinyVocoderModel(Path)}: 1,282-dim spectral vocoder GGUF v3 model
 *       ({@code tiny-vocoder.gguf})</li>
 *   <li>{@link #generateTestPng(Path, String)}: 16x16 RGB PNG image</li>
 *   <li>{@link #generateTestWav16kHz(Path, String)}: 16kHz 16-bit mono PCM RIFF WAV audio</li>
 * </ul>
 */
public final class ArgusSyntheticTestFixtures {

    private ArgusSyntheticTestFixtures() {}

    /**
     * Resolves the repository root directory containing {@code README.md} and {@code tests/data/tiny.gguf}.
     */
    public static Path resolveRepoRoot() {
        Path current = Path.of("").toAbsolutePath().normalize();
        for (int i = 0; i < 6 && current != null; i++) {
            if (Files.isRegularFile(current.resolve("tests/data/tiny.gguf"))
                    && Files.isRegularFile(current.resolve("README.md"))) {
                return current;
            }
            current = current.getParent();
        }
        throw new IllegalStateException("Could not locate libargus repository root from " + Path.of("").toAbsolutePath());
    }

    /**
     * Generates a minimal valid Whisper GGML model ({@code tiny-whisper.bin}) that executes
     * end-to-end on CPU in milliseconds and deterministically transcribes {@code " argus"}.
     */
    public static Path generateTinyWhisperModel(Path targetDir) throws IOException {
        Path out = targetDir.resolve("tiny-whisper.bin");
        if (Files.isRegularFile(out)) {
            return out;
        }

        final int nVocab = 51864;
        final int nAudioCtx = 64;
        final int nAudioState = 8;
        final int nAudioHead = 2;
        final int nAudioLayer = 1;
        final int nTextCtx = 16;
        final int nTextState = 8;
        final int nTextHead = 2;
        // n_text_layer == 2 && n_vocab != 51866 activates Whisper's single-pass no_timestamps fast path
        final int nTextLayer = 2;
        final int nMels = 80;
        final int ftype = 0; // GGML_TYPE_F32
        final int nFft = 201;

        ByteArrayOutputStream bos = new ByteArrayOutputStream(2 * 1024 * 1024);

        // 1. Magic + 11 int32 hparams + mel filter header
        ByteBuffer hdr = ByteBuffer.allocate(14 * 4).order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(0x67676d6c); // "ggml"
        hdr.putInt(nVocab);
        hdr.putInt(nAudioCtx);
        hdr.putInt(nAudioState);
        hdr.putInt(nAudioHead);
        hdr.putInt(nAudioLayer);
        hdr.putInt(nTextCtx);
        hdr.putInt(nTextState);
        hdr.putInt(nTextHead);
        hdr.putInt(nTextLayer);
        hdr.putInt(nMels);
        hdr.putInt(ftype);
        hdr.putInt(nMels);
        hdr.putInt(nFft);
        bos.write(hdr.array());

        // 2. Mel filterbank weights (nMels * nFft floats)
        ByteBuffer melBuf = ByteBuffer.allocate(nMels * nFft * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < nMels * nFft; i++) {
            melBuf.putFloat(0.01f);
        }
        bos.write(melBuf.array());

        // 3. Stored vocabulary tokens (whisper_model_load auto-populates special tokens 2..51863)
        // Token 0 MUST be " " because suppress_blank looks up vocab.token_to_id.at(" ").
        // Token 1 is " argus".
        byte[] tok0 = " ".getBytes(StandardCharsets.UTF_8);
        byte[] tok1 = " argus".getBytes(StandardCharsets.UTF_8);
        ByteBuffer vocabBuf = ByteBuffer.allocate(4 + 4 + tok0.length + 4 + tok1.length)
                .order(ByteOrder.LITTLE_ENDIAN);
        vocabBuf.putInt(2);
        vocabBuf.putInt(tok0.length);
        vocabBuf.put(tok0);
        vocabBuf.putInt(tok1.length);
        vocabBuf.put(tok1);
        bos.write(vocabBuf.array());

        // 4. Encoder & decoder tensors
        // Encoder base (7)
        writeWhisperTensor(bos, "encoder.positional_embedding", new int[]{nAudioState, nAudioCtx}, null);
        writeWhisperTensor(bos, "encoder.conv1.weight", new int[]{3, nMels, nAudioState}, null);
        writeWhisperTensor(bos, "encoder.conv1.bias", new int[]{1, nAudioState}, null);
        writeWhisperTensor(bos, "encoder.conv2.weight", new int[]{3, nAudioState, nAudioState}, null);
        writeWhisperTensor(bos, "encoder.conv2.bias", new int[]{1, nAudioState}, null);
        writeWhisperTensor(bos, "encoder.ln_post.weight", new int[]{nAudioState}, null);
        writeWhisperTensor(bos, "encoder.ln_post.bias", new int[]{nAudioState}, null);

        // Encoder blocks
        for (int i = 0; i < nAudioLayer; i++) {
            String p = "encoder.blocks." + i + ".";
            writeWhisperTensor(bos, p + "mlp_ln.weight", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "mlp_ln.bias", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "mlp.0.weight", new int[]{nAudioState, 4 * nAudioState}, null);
            writeWhisperTensor(bos, p + "mlp.0.bias", new int[]{4 * nAudioState}, null);
            writeWhisperTensor(bos, p + "mlp.2.weight", new int[]{4 * nAudioState, nAudioState}, null);
            writeWhisperTensor(bos, p + "mlp.2.bias", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "attn_ln.weight", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "attn_ln.bias", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.query.weight", new int[]{nAudioState, nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.query.bias", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.key.weight", new int[]{nAudioState, nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.value.weight", new int[]{nAudioState, nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.value.bias", new int[]{nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.out.weight", new int[]{nAudioState, nAudioState}, null);
            writeWhisperTensor(bos, p + "attn.out.bias", new int[]{nAudioState}, null);
        }

        // Decoder base (4)
        writeWhisperTensor(bos, "decoder.positional_embedding", new int[]{nTextState, nTextCtx}, null);

        // Construct decoder.token_embedding.weight so that:
        // - Step 0 (is_initial = true, token_eot=50256 masked to -inf) selects token 1 (" argus")
        // - Step 1 (is_initial = false, token_eot=50256 unmasked) selects token_eot (50256)
        float[] dTe = new float[nTextState * nVocab];
        for (int tok = 0; tok < nVocab; tok++) {
            dTe[tok * nTextState] = -10.0f;
        }
        dTe[1 * nTextState] = 15.0f;      // token 1 (" argus")
        dTe[50256 * nTextState] = 20.0f;  // token_eot ("<|endoftext|>")
        writeWhisperTensor(bos, "decoder.token_embedding.weight", new int[]{nTextState, nVocab}, dTe);

        writeWhisperTensor(bos, "decoder.ln.weight", new int[]{nTextState}, null);
        float[] dLnBias = new float[nTextState];
        dLnBias[0] = 1.0f;
        writeWhisperTensor(bos, "decoder.ln.bias", new int[]{nTextState}, dLnBias);

        // Decoder blocks
        for (int i = 0; i < nTextLayer; i++) {
            String p = "decoder.blocks." + i + ".";
            writeWhisperTensor(bos, p + "mlp_ln.weight", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "mlp_ln.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "mlp.0.weight", new int[]{nTextState, 4 * nTextState}, null);
            writeWhisperTensor(bos, p + "mlp.0.bias", new int[]{4 * nTextState}, null);
            writeWhisperTensor(bos, p + "mlp.2.weight", new int[]{4 * nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "mlp.2.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "attn_ln.weight", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "attn_ln.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "attn.query.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "attn.query.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "attn.key.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "attn.value.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "attn.value.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "attn.out.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "attn.out.bias", new int[]{nTextState}, null);

            writeWhisperTensor(bos, p + "cross_attn_ln.weight", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn_ln.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.query.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.query.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.key.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.value.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.value.bias", new int[]{nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.out.weight", new int[]{nTextState, nTextState}, null);
            writeWhisperTensor(bos, p + "cross_attn.out.bias", new int[]{nTextState}, null);
        }

        Files.write(out, bos.toByteArray());
        return out;
    }

    private static void writeWhisperTensor(ByteArrayOutputStream bos, String name, int[] shape, float[] values)
            throws IOException {
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        int nDims = shape.length;
        int nElements = 1;
        for (int d : shape) {
            nElements *= d;
        }
        ByteBuffer hdr = ByteBuffer.allocate((3 + nDims) * 4 + nameBytes.length + nElements * 4)
                .order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(nDims);
        hdr.putInt(nameBytes.length);
        hdr.putInt(0); // GGML_TYPE_F32
        for (int d : shape) {
            hdr.putInt(d);
        }
        hdr.put(nameBytes);
        if (values != null) {
            for (float v : values) {
                hdr.putFloat(v);
            }
        } else {
            hdr.position(hdr.position() + nElements * 4);
        }
        bos.write(hdr.array());
    }

    /**
     * Generates a unified Vision ({@code mlp}) + Audio ({@code gemma4ua}) GGUF v3 projector
     * ({@code tiny-mmproj-omni.gguf}) compatible with {@code tests/data/tiny.gguf} (32-dim).
     */
    public static Path generateTinyOmniProjector(Path targetDir) throws IOException {
        Path out = targetDir.resolve("tiny-mmproj-omni.gguf");
        if (Files.isRegularFile(out)) {
            return out;
        }

        GgufV3Builder b = new GgufV3Builder();
        b.addString("general.architecture", "clip");
        b.addString("general.name", "Argus synthetic omni projector");
        b.addUint32("general.file_type", 0);
        b.addBool("clip.has_vision_encoder", true);
        b.addBool("clip.has_audio_encoder", true);
        b.addBool("clip.use_gelu", true);

        // Vision encoder metadata (matches scripts/export_test_projector.py)
        b.addString("clip.vision.projector_type", "mlp");
        b.addUint32("clip.vision.embedding_length", 8);
        b.addUint32("clip.vision.feed_forward_length", 16);
        b.addUint32("clip.vision.block_count", 2);
        b.addUint32("clip.vision.projection_dim", 32);
        b.addUint32("clip.vision.attention.head_count", 2);
        b.addUint32("clip.vision.image_size", 16);
        b.addUint32("clip.vision.patch_size", 8);
        b.addFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
        b.addFloat32Array("clip.vision.image_mean", new float[]{0.5f, 0.5f, 0.5f});
        b.addFloat32Array("clip.vision.image_std", new float[]{0.5f, 0.5f, 0.5f});

        // Audio encoder metadata (encoder-free gemma4ua raw-waveform projector)
        b.addString("clip.audio.projector_type", "gemma4ua");
        b.addUint32("clip.audio.embedding_length", 640);
        b.addUint32("clip.audio.feed_forward_length", 16);
        b.addUint32("clip.audio.block_count", 0);
        b.addUint32("clip.audio.projection_dim", 32);
        b.addUint32("clip.audio.attention.head_count", 2);
        b.addUint32("clip.audio.num_mel_bins", 640);
        b.addFloat32("clip.audio.attention.layer_norm_epsilon", 1e-6f);

        // Vision tensors (24 tensors)
        b.addHashSeededTensor("v.patch_embd.weight", new long[]{8, 8, 3, 8}, false);
        b.addHashSeededTensor("v.position_embd.weight", new long[]{8, 4}, false);
        for (int layer = 0; layer < 2; layer++) {
            String prefix = "v.blk." + layer + ".";
            b.addHashSeededTensor(prefix + "attn_q.weight", new long[]{8, 8}, false);
            b.addHashSeededTensor(prefix + "attn_k.weight", new long[]{8, 8}, false);
            b.addHashSeededTensor(prefix + "attn_v.weight", new long[]{8, 8}, false);
            b.addHashSeededTensor(prefix + "attn_out.weight", new long[]{8, 8}, false);
            b.addHashSeededTensor(prefix + "ln1.weight", new long[]{8}, true);
            b.addHashSeededTensor(prefix + "ln2.weight", new long[]{8}, true);
            b.addHashSeededTensor(prefix + "ffn_up.weight", new long[]{8, 16}, false);
            b.addHashSeededTensor(prefix + "ffn_down.weight", new long[]{16, 8}, false);
        }
        b.addHashSeededTensor("mm.0.weight", new long[]{8, 32}, false);
        b.addHashSeededTensor("mm.0.bias", new long[]{32}, false);
        b.addHashSeededTensor("mm.2.weight", new long[]{32, 32}, false);
        b.addHashSeededTensor("mm.2.bias", new long[]{32}, false);

        // Audio tensor (1 tensor: 640 -> 32 projection)
        b.addHashSeededTensor("mm.a.input_projection.weight", new long[]{640, 32}, false);

        Files.write(out, b.build());
        return out;
    }

    /**
     * Generates a 1-layer, 32-dim causal LLaMA GGUF v3 model ({@code tiny-causal-260.gguf}) with
     * full 256-byte fallback vocabulary ({@code <0x00>..<0xFF>}) so universal UTF-8 prompts,
     * multimodal {@code mtmd_tokenize}, KV shifting, and TTS speech synthesis all execute cleanly.
     */
    public static Path generateTinyCausalModel(Path targetDir) throws IOException {
        Path out = targetDir.resolve("tiny-causal-260.gguf");
        if (Files.isRegularFile(out)) {
            return out;
        }

        final String arch = "llama";
        final int nEmbd = 32;
        final int nHead = 2;
        final int nHeadKv = 2;
        final int nLayer = 1;
        final int nFf = 64;
        final int nCtx = 512;
        final int vocabSize = 260;

        GgufV3Builder b = new GgufV3Builder();
        b.addString("general.architecture", arch);
        b.addUint32(arch + ".context_length", nCtx);
        b.addUint32(arch + ".embedding_length", nEmbd);
        b.addUint32(arch + ".block_count", nLayer);
        b.addUint32(arch + ".feed_forward_length", nFf);
        b.addUint32(arch + ".attention.head_count", nHead);
        b.addUint32(arch + ".attention.head_count_kv", nHeadKv);
        b.addUint32(arch + ".rope.dimension_count", nEmbd / nHead);
        b.addFloat32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);

        b.addString("tokenizer.ggml.model", "llama");
        List<String> tokens = new ArrayList<>(vocabSize);
        tokens.add("<unk>");
        tokens.add("<s>");
        tokens.add("</s>");
        tokens.add("<pad>");
        int[] tokTypes = new int[vocabSize];
        tokTypes[0] = 2; // UNKNOWN
        tokTypes[1] = 3; // CONTROL
        tokTypes[2] = 3; // CONTROL
        tokTypes[3] = 3; // CONTROL
        float[] scores = new float[vocabSize];
        for (int i = 0; i < 256; i++) {
            tokens.add(String.format("<0x%02X>", i));
            tokTypes[4 + i] = 6; // BYTE
        }
        b.addStringArray("tokenizer.ggml.tokens", tokens);
        b.addFloat32Array("tokenizer.ggml.scores", scores);
        b.addInt32Array("tokenizer.ggml.token_type", tokTypes);
        b.addUint32("tokenizer.ggml.bos_token_id", 1);
        b.addUint32("tokenizer.ggml.eos_token_id", 2);
        b.addUint32("tokenizer.ggml.padding_token_id", 3);

        b.addHashSeededTensor("token_embd.weight", new long[]{nEmbd, vocabSize}, false);
        b.addHashSeededTensor("output_norm.weight", new long[]{nEmbd}, true);
        b.addHashSeededTensor("output.weight", new long[]{nEmbd, vocabSize}, false);

        b.addHashSeededTensor("blk.0.attn_norm.weight", new long[]{nEmbd}, true);
        b.addHashSeededTensor("blk.0.attn_q.weight", new long[]{nEmbd, nEmbd}, false);
        b.addHashSeededTensor("blk.0.attn_k.weight", new long[]{nEmbd, nEmbd}, false);
        b.addHashSeededTensor("blk.0.attn_v.weight", new long[]{nEmbd, nEmbd}, false);
        b.addHashSeededTensor("blk.0.attn_output.weight", new long[]{nEmbd, nEmbd}, false);
        b.addHashSeededTensor("blk.0.ffn_norm.weight", new long[]{nEmbd}, true);
        b.addHashSeededTensor("blk.0.ffn_gate.weight", new long[]{nEmbd, nFf}, false);
        b.addHashSeededTensor("blk.0.ffn_up.weight", new long[]{nEmbd, nFf}, false);
        b.addHashSeededTensor("blk.0.ffn_down.weight", new long[]{nFf, nEmbd}, false);

        Files.write(out, b.build());
        return out;
    }

    /**
     * Generates a 1,282-dimensional spectral vocoder GGUF v3 model ({@code tiny-vocoder.gguf})
     * using the {@code gemma-embedding2} decoupled output projection architecture
     * ({@code n_embd = 32}, {@code n_embd_out = 1282}) for end-to-end {@code synthesizeSpeech} tests.
     */
    public static Path generateTinyVocoderModel(Path targetDir) throws IOException {
        Path out = targetDir.resolve("tiny-vocoder.gguf");
        if (Files.isRegularFile(out)) {
            return out;
        }

        final String arch = "gemma-embedding2";
        final int nEmbd = 32;
        final int nEmbdOut = 1282; // 2 * (1280 / 2 + 1) spectral bins for overlap-add iSTFT
        final int nEmbdPerLayer = 8;
        final int nHead = 2;
        final int nHeadKv = 2;
        final int nEmbdHead = nEmbd / nHead;
        final int nLayer = 2;
        final int nFf = 64;
        final int nCtx = 512;
        final int nSwa = 64;
        final int vocabSize = 260;

        GgufV3Builder b = new GgufV3Builder();
        b.addString("general.architecture", arch);
        b.addUint32(arch + ".context_length", nCtx);
        b.addUint32(arch + ".embedding_length", nEmbd);
        b.addUint32(arch + ".embedding_length_out", nEmbdOut);
        b.addUint32(arch + ".embedding_length_per_layer_input", nEmbdPerLayer);
        b.addUint32(arch + ".block_count", nLayer);
        b.addUint32(arch + ".feed_forward_length", nFf);
        b.addUint32(arch + ".attention.head_count", nHead);
        b.addUint32(arch + ".attention.head_count_kv", nHeadKv);
        b.addUint32(arch + ".attention.key_length", nEmbdHead);
        b.addUint32(arch + ".attention.value_length", nEmbdHead);
        b.addUint32(arch + ".attention.key_length_swa", nEmbdHead);
        b.addUint32(arch + ".attention.value_length_swa", nEmbdHead);
        b.addUint32(arch + ".attention.sliding_window", nSwa);
        b.addUint32(arch + ".attention.sliding_window_pattern", 2);
        b.addUint32(arch + ".rope.dimension_count", nEmbdHead);
        b.addUint32(arch + ".rope.dimension_count_swa", nEmbdHead);
        b.addFloat32(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        b.addUint32(arch + ".pooling_type", 0); // LLAMA_POOLING_TYPE_NONE for per-token spectral frames
        b.addBool(arch + ".attention.causal", false);

        b.addString("tokenizer.ggml.model", "llama");
        List<String> tokens = new ArrayList<>(vocabSize);
        tokens.add("<unk>");
        tokens.add("<s>");
        tokens.add("</s>");
        tokens.add("<pad>");
        int[] tokTypes = new int[vocabSize];
        tokTypes[0] = 2; // UNKNOWN
        tokTypes[1] = 3; // CONTROL
        tokTypes[2] = 3; // CONTROL
        tokTypes[3] = 3; // CONTROL
        float[] scores = new float[vocabSize];
        for (int i = 0; i < 256; i++) {
            tokens.add(String.format("<0x%02X>", i));
            tokTypes[4 + i] = 6; // BYTE
        }
        b.addStringArray("tokenizer.ggml.tokens", tokens);
        b.addFloat32Array("tokenizer.ggml.scores", scores);
        b.addInt32Array("tokenizer.ggml.token_type", tokTypes);
        b.addUint32("tokenizer.ggml.bos_token_id", 1);
        b.addUint32("tokenizer.ggml.eos_token_id", 2);
        b.addUint32("tokenizer.ggml.padding_token_id", 3);

        // Tensors (shapes in ggml order: {ne0, ne1})
        b.addHashSeededTensor("token_embd.weight", new long[]{nEmbd, vocabSize}, false);
        b.addHashSeededTensor("per_layer_model_proj.weight", new long[]{nEmbd, (long) nEmbdPerLayer * nLayer}, false);
        b.addHashSeededTensor("per_layer_proj_norm.weight", new long[]{nEmbdPerLayer}, true);
        b.addHashSeededTensor("output_norm.weight", new long[]{nEmbd}, true);
        b.addHashSeededTensor("output.weight", new long[]{nEmbd, nEmbdOut}, false);

        for (int il = 0; il < nLayer; il++) {
            String p = "blk." + il + ".";
            b.addHashSeededTensor(p + "attn_norm.weight", new long[]{nEmbd}, true);
            b.addHashSeededTensor(p + "attn_q.weight", new long[]{nEmbd, nEmbd}, false);
            b.addHashSeededTensor(p + "attn_k.weight", new long[]{nEmbd, nEmbd}, false);
            b.addHashSeededTensor(p + "attn_v.weight", new long[]{nEmbd, nEmbd}, false);
            b.addHashSeededTensor(p + "attn_output.weight", new long[]{nEmbd, nEmbd}, false);
            b.addHashSeededTensor(p + "attn_q_norm.weight", new long[]{nEmbdHead}, true);
            b.addHashSeededTensor(p + "attn_k_norm.weight", new long[]{nEmbdHead}, true);
            b.addHashSeededTensor(p + "post_attention_norm.weight", new long[]{nEmbd}, true);

            b.addHashSeededTensor(p + "ffn_norm.weight", new long[]{nEmbd}, true);
            b.addHashSeededTensor(p + "ffn_gate.weight", new long[]{nEmbd, nFf}, false);
            b.addHashSeededTensor(p + "ffn_up.weight", new long[]{nEmbd, nFf}, false);
            b.addHashSeededTensor(p + "ffn_down.weight", new long[]{nFf, nEmbd}, false);
            b.addHashSeededTensor(p + "post_ffw_norm.weight", new long[]{nEmbd}, true);

            b.addHashSeededTensor(p + "inp_gate.weight", new long[]{nEmbd, nEmbdPerLayer}, false);
            b.addHashSeededTensor(p + "proj.weight", new long[]{nEmbdPerLayer, nEmbd}, false);
            b.addHashSeededTensor(p + "post_norm.weight", new long[]{nEmbd}, true);
            b.addHashSeededTensor(p + "layer_output_scale.weight", new long[]{1}, true);
        }

        Files.write(out, b.build());
        return out;
    }

    /**
     * Generates a valid 16x16 RGB PNG image file for vision and multimodal tests.
     */
    public static Path generateTestPng(Path targetDir, String filename) throws IOException {
        Path out = targetDir.resolve(filename);
        if (Files.isRegularFile(out)) {
            return out;
        }
        BufferedImage img = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                int r = (x * 16) & 0xFF;
                int g = (y * 16) & 0xFF;
                int b = ((x + y) * 8) & 0xFF;
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        ImageIO.write(img, "png", out.toFile());
        return out;
    }

    /**
     * Generates a valid 1-second 16kHz 16-bit mono PCM RIFF WAV file for ASR and multimodal audio tests.
     */
    public static Path generateTestWav16kHz(Path targetDir, String filename) throws IOException {
        Path out = targetDir.resolve(filename);
        if (Files.isRegularFile(out)) {
            return out;
        }
        final int sampleRate = 16000;
        final int numSamples = 16000; // 1.0 second
        final int dataBytes = numSamples * 2;
        ByteBuffer buf = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        buf.put("RIFF".getBytes(StandardCharsets.US_ASCII));
        buf.putInt(36 + dataBytes);
        buf.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        buf.put("fmt ".getBytes(StandardCharsets.US_ASCII));
        buf.putInt(16); // PCM chunk size
        buf.putShort((short) 1); // AudioFormat = 1 (PCM)
        buf.putShort((short) 1); // NumChannels = 1
        buf.putInt(sampleRate);
        buf.putInt(sampleRate * 2); // ByteRate
        buf.putShort((short) 2); // BlockAlign
        buf.putShort((short) 16); // BitsPerSample
        buf.put("data".getBytes(StandardCharsets.US_ASCII));
        buf.putInt(dataBytes);
        for (int i = 0; i < numSamples; i++) {
            double t = (double) i / sampleRate;
            short sample = (short) (Math.sin(2.0 * Math.PI * 440.0 * t) * 8192.0);
            buf.putShort(sample);
        }
        Files.write(out, buf.array());
        return out;
    }

    /**
     * Minimal, zero-dependency GGUF v3 binary builder.
     */
    private static final class GgufV3Builder {
        private final List<byte[]> metadata = new ArrayList<>();
        private final List<TensorEntry> tensors = new ArrayList<>();

        private record TensorEntry(String name, long[] shape, byte[] data) {}

        private static byte[] encodeString(String s) {
            byte[] utf8 = s.getBytes(StandardCharsets.UTF_8);
            ByteBuffer buf = ByteBuffer.allocate(8 + utf8.length).order(ByteOrder.LITTLE_ENDIAN);
            buf.putLong(utf8.length);
            buf.put(utf8);
            return buf.array();
        }

        void addUint32(String key, int value) {
            byte[] k = encodeString(key);
            ByteBuffer buf = ByteBuffer.allocate(k.length + 8).order(ByteOrder.LITTLE_ENDIAN);
            buf.put(k);
            buf.putInt(4); // GGUF_TYPE_UINT32
            buf.putInt(value);
            metadata.add(buf.array());
        }

        void addFloat32(String key, float value) {
            byte[] k = encodeString(key);
            ByteBuffer buf = ByteBuffer.allocate(k.length + 8).order(ByteOrder.LITTLE_ENDIAN);
            buf.put(k);
            buf.putInt(6); // GGUF_TYPE_FLOAT32
            buf.putFloat(value);
            metadata.add(buf.array());
        }

        void addBool(String key, boolean value) {
            byte[] k = encodeString(key);
            ByteBuffer buf = ByteBuffer.allocate(k.length + 5).order(ByteOrder.LITTLE_ENDIAN);
            buf.put(k);
            buf.putInt(7); // GGUF_TYPE_BOOL
            buf.put((byte) (value ? 1 : 0));
            metadata.add(buf.array());
        }

        void addString(String key, String value) {
            byte[] k = encodeString(key);
            byte[] v = encodeString(value);
            ByteBuffer buf = ByteBuffer.allocate(k.length + 4 + v.length).order(ByteOrder.LITTLE_ENDIAN);
            buf.put(k);
            buf.putInt(8); // GGUF_TYPE_STRING
            buf.put(v);
            metadata.add(buf.array());
        }

        void addFloat32Array(String key, float[] values) {
            byte[] k = encodeString(key);
            ByteBuffer buf = ByteBuffer.allocate(k.length + 16 + values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            buf.put(k);
            buf.putInt(9); // GGUF_TYPE_ARRAY
            buf.putInt(6); // GGUF_TYPE_FLOAT32
            buf.putLong(values.length);
            for (float v : values) {
                buf.putFloat(v);
            }
            metadata.add(buf.array());
        }

        void addInt32Array(String key, int[] values) {
            byte[] k = encodeString(key);
            ByteBuffer buf = ByteBuffer.allocate(k.length + 16 + values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            buf.put(k);
            buf.putInt(9); // GGUF_TYPE_ARRAY
            buf.putInt(5); // GGUF_TYPE_INT32
            buf.putLong(values.length);
            for (int v : values) {
                buf.putInt(v);
            }
            metadata.add(buf.array());
        }

        void addStringArray(String key, List<String> values) {
            ByteArrayOutputStream arrBos = new ByteArrayOutputStream();
            try {
                arrBos.write(encodeString(key));
                ByteBuffer hdr = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
                hdr.putInt(9); // GGUF_TYPE_ARRAY
                hdr.putInt(8); // GGUF_TYPE_STRING
                hdr.putLong(values.size());
                arrBos.write(hdr.array());
                for (String s : values) {
                    arrBos.write(encodeString(s));
                }
            } catch (IOException e) {
                throw new AssertionError(e);
            }
            metadata.add(arrBos.toByteArray());
        }

        void addHashSeededTensor(String name, long[] shape, boolean normOnes) {
            int count = 1;
            for (long d : shape) {
                count = Math.multiplyExact(count, Math.toIntExact(d));
            }
            int seed = sha256PrefixInt(name);
            ByteBuffer data = ByteBuffer.allocate(count * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < count; i++) {
                float val = normOnes ? 1.0f : ((((seed + i * 2654435761L) & 255L) - 128L) / 8192.0f);
                data.putFloat(val);
            }
            tensors.add(new TensorEntry(name, shape, data.array()));
        }

        private static int sha256PrefixInt(String s) {
            try {
                byte[] digest = MessageDigest.getInstance("SHA-256")
                        .digest(s.getBytes(StandardCharsets.UTF_8));
                return ByteBuffer.wrap(digest, 0, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            } catch (NoSuchAlgorithmException e) {
                throw new AssertionError(e);
            }
        }

        byte[] build() throws IOException {
            ByteArrayOutputStream headerBos = new ByteArrayOutputStream();
            headerBos.write("GGUF".getBytes(StandardCharsets.US_ASCII));
            ByteBuffer counts = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
            counts.putInt(3); // GGUF v3
            counts.putLong(tensors.size());
            counts.putLong(metadata.size());
            headerBos.write(counts.array());

            for (byte[] kv : metadata) {
                headerBos.write(kv);
            }

            ByteArrayOutputStream payloadBos = new ByteArrayOutputStream();
            long offset = 0;
            for (TensorEntry t : tensors) {
                headerBos.write(encodeString(t.name()));
                ByteBuffer thdr = ByteBuffer.allocate(4 + t.shape().length * 8 + 12)
                        .order(ByteOrder.LITTLE_ENDIAN);
                thdr.putInt(t.shape().length);
                for (long d : t.shape()) {
                    thdr.putLong(d);
                }
                thdr.putInt(0); // GGML_TYPE_F32
                thdr.putLong(offset);
                headerBos.write(thdr.array());

                byte[] data = t.data();
                payloadBos.write(data);
                int pad = (int) ((-data.length) & 31);
                if (pad > 0) {
                    payloadBos.write(new byte[pad]);
                }
                offset += data.length + pad;
            }

            byte[] headerBytes = headerBos.toByteArray();
            int headerPad = (-headerBytes.length) & 31;
            ByteArrayOutputStream full = new ByteArrayOutputStream(
                    headerBytes.length + headerPad + payloadBos.size());
            full.write(headerBytes);
            if (headerPad > 0) {
                full.write(new byte[headerPad]);
            }
            payloadBos.writeTo(full);
            return full.toByteArray();
        }
    }
}
