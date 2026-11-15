package cc.projectargus.libargus.langchain4j;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Objects;

/**
 * Pure-Java, zero-dependency, Android API 26+-compatible RIFF/WAVE and raw PCM audio codec.
 *
 * <p>Avoids {@code javax.sound.sampled.AudioSystem} ({@code java.desktop}) so
 * {@code :libargus-langchain4j} executes cleanly on Android, minimal JLink server images,
 * and standard desktop/server JVMs.
 */
final class LibArgusAudioCodec {

    static final int WHISPER_SAMPLE_RATE_HZ = 16_000;
    static final int DEFAULT_TTS_SAMPLE_RATE_HZ = 24_000;

    private static final int WAVE_FORMAT_PCM = 0x0001;
    private static final int WAVE_FORMAT_IEEE_FLOAT = 0x0003;
    private static final int WAVE_FORMAT_EXTENSIBLE = 0xFFFE;

    private LibArgusAudioCodec() {}

    /**
     * Decodes a RIFF/WAVE container (8/16/24/32-bit PCM or 32-bit IEEE float, mono or multi-channel)
     * or raw 16kHz PCM payload ({@code audio/L16}, {@code audio/pcm}, {@code audio/f32le}) into
     * normalized 16 kHz mono {@code float[]} samples in {@code [-1.0f, 1.0f]}.
     */
    static float[] decodeToMono16kHzPcm(byte[] audioBytes, String mimeType, long maxMediaBytes) {
        Objects.requireNonNull(audioBytes, "audioBytes");
        if (audioBytes.length == 0) {
            throw new IllegalArgumentException("Audio payload cannot be empty (0 bytes)");
        }
        if (audioBytes.length > maxMediaBytes) {
            throw new IllegalArgumentException(
                "Audio payload (" + audioBytes.length + " bytes) exceeds maxMediaBytes limit (" + maxMediaBytes + " bytes)"
            );
        }

        if (isRiffWave(audioBytes)) {
            return decodeRiffWaveTo16kHzMono(audioBytes, maxMediaBytes);
        }

        String lowerMime = mimeType != null ? mimeType.toLowerCase(Locale.ROOT).trim() : "";
        if (lowerMime.startsWith("audio/f32le")) {
            if ((audioBytes.length & 3) != 0) {
                throw new IllegalArgumentException("audio/f32le payload byte length (" + audioBytes.length + ") must be a multiple of 4");
            }
            int nSamples = audioBytes.length / 4;
            checkDecodedSampleLimit(nSamples, maxMediaBytes);
            ByteBuffer buf = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN);
            float[] out = new float[nSamples];
            for (int i = 0; i < nSamples; i++) {
                float f = buf.getFloat();
                out[i] = Float.isFinite(f) ? clampUnit(f) : 0.0f;
            }
            return out;
        }

        if (lowerMime.startsWith("audio/l16") || lowerMime.startsWith("audio/pcm")) {
            if ((audioBytes.length & 1) != 0) {
                throw new IllegalArgumentException(lowerMime + " payload byte length (" + audioBytes.length + ") must be even");
            }
            int nSamples = audioBytes.length / 2;
            checkDecodedSampleLimit(nSamples, maxMediaBytes);
            ByteBuffer buf = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN);
            float[] out = new float[nSamples];
            for (int i = 0; i < nSamples; i++) {
                out[i] = buf.getShort() / 32768.0f;
            }
            return out;
        }

        throw new IllegalArgumentException(
            "Unsupported audio format (expected RIFF/WAVE container or raw audio/L16, audio/pcm, audio/f32le; mimeType="
                + (mimeType != null ? mimeType : "unspecified") + ")"
        );
    }

    /**
     * Encodes normalized {@code [-1.0f, 1.0f]} mono PCM float samples into a canonical 44-byte-header
     * 16-bit signed little-endian RIFF/WAVE byte array ({@code audio/wav}).
     */
    static byte[] encodeMonoPcm16Wav(float[] pcmSamples, int sampleRate) {
        Objects.requireNonNull(pcmSamples, "pcmSamples");
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be > 0 (got " + sampleRate + ")");
        }
        int dataBytes = Math.multiplyExact(pcmSamples.length, 2);
        int totalBytes = Math.addExact(44, dataBytes);
        int byteRate = Math.multiplyExact(sampleRate, 2);

        ByteBuffer buf = ByteBuffer.allocate(totalBytes).order(ByteOrder.LITTLE_ENDIAN);
        buf.put((byte) 'R').put((byte) 'I').put((byte) 'F').put((byte) 'F');
        buf.putInt(36 + dataBytes);
        buf.put((byte) 'W').put((byte) 'A').put((byte) 'V').put((byte) 'E');
        buf.put((byte) 'f').put((byte) 'm').put((byte) 't').put((byte) ' ');
        buf.putInt(16);                // PCM fmt chunk size
        buf.putShort((short) 1);       // AudioFormat = WAVE_FORMAT_PCM
        buf.putShort((short) 1);       // NumChannels = 1 (mono)
        buf.putInt(sampleRate);        // SampleRate
        buf.putInt(byteRate);          // ByteRate = sampleRate * 1 * 2
        buf.putShort((short) 2);       // BlockAlign = 2 bytes
        buf.putShort((short) 16);      // BitsPerSample = 16
        buf.put((byte) 'd').put((byte) 'a').put((byte) 't').put((byte) 'a');
        buf.putInt(dataBytes);

        for (float sample : pcmSamples) {
            float clean = Float.isFinite(sample) ? clampUnit(sample) : 0.0f;
            int quantized = Math.round(clean * 32767.0f);
            if (quantized > Short.MAX_VALUE) quantized = Short.MAX_VALUE;
            if (quantized < Short.MIN_VALUE) quantized = Short.MIN_VALUE;
            buf.putShort((short) quantized);
        }
        return buf.array();
    }

    private static boolean isRiffWave(byte[] b) {
        return b.length >= 44
            && b[0] == 'R' && b[1] == 'I' && b[2] == 'F' && b[3] == 'F'
            && b[8] == 'W' && b[9] == 'A' && b[10] == 'V' && b[11] == 'E';
    }

    private static float[] decodeRiffWaveTo16kHzMono(byte[] bytes, long maxMediaBytes) {
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int pos = 12;

        int audioFormat = -1;
        int channels = -1;
        int sampleRate = -1;
        int blockAlign = -1;
        int bitsPerSample = -1;
        int dataOffset = -1;
        int dataLength = -1;

        while (pos + 8 <= bytes.length) {
            String chunkId = new String(bytes, pos, 4, StandardCharsets.US_ASCII);
            long rawChunkSize = Integer.toUnsignedLong(buf.getInt(pos + 4));
            int chunkBodyStart = pos + 8;
            long remaining = bytes.length - chunkBodyStart;

            if ("fmt ".equals(chunkId)) {
                if (rawChunkSize < 16 || rawChunkSize > remaining) {
                    throw new IllegalArgumentException("Malformed WAV fmt chunk");
                }
                int rawFormat = Short.toUnsignedInt(buf.getShort(chunkBodyStart));
                channels = Short.toUnsignedInt(buf.getShort(chunkBodyStart + 2));
                sampleRate = buf.getInt(chunkBodyStart + 4);
                blockAlign = Short.toUnsignedInt(buf.getShort(chunkBodyStart + 12));
                bitsPerSample = Short.toUnsignedInt(buf.getShort(chunkBodyStart + 14));

                if (rawFormat == WAVE_FORMAT_EXTENSIBLE) {
                    if (rawChunkSize < 40) {
                        throw new IllegalArgumentException("Malformed WAVE_FORMAT_EXTENSIBLE fmt chunk");
                    }
                    audioFormat = Short.toUnsignedInt(buf.getShort(chunkBodyStart + 24));
                } else {
                    audioFormat = rawFormat;
                }
            } else if ("data".equals(chunkId)) {
                dataOffset = chunkBodyStart;
                dataLength = (int) Math.min(rawChunkSize, remaining);
                break;
            }

            long paddedAdvance = rawChunkSize + (rawChunkSize & 1L);
            if (paddedAdvance > remaining) {
                break;
            }
            pos = (int) (chunkBodyStart + paddedAdvance);
        }

        if (audioFormat < 0 || dataOffset < 0) {
            throw new IllegalArgumentException("Malformed WAV container: missing fmt or data chunk");
        }
        if (audioFormat != WAVE_FORMAT_PCM && audioFormat != WAVE_FORMAT_IEEE_FLOAT) {
            throw new IllegalArgumentException("Unsupported WAV encoding format code: " + audioFormat);
        }
        if (channels < 1 || channels > 16) {
            throw new IllegalArgumentException("Unsupported WAV channel count: " + channels);
        }
        if (sampleRate < 1_000 || sampleRate > 384_000) {
            throw new IllegalArgumentException("Unsupported WAV sample rate: " + sampleRate + " Hz");
        }
        int bytesPerSample = bitsPerSample / 8;
        if (bytesPerSample < 1 || (bitsPerSample != 8 && bitsPerSample != 16 && bitsPerSample != 24 && bitsPerSample != 32)) {
            throw new IllegalArgumentException("Unsupported WAV bit depth: " + bitsPerSample);
        }
        if (audioFormat == WAVE_FORMAT_IEEE_FLOAT && bitsPerSample != 32) {
            throw new IllegalArgumentException("IEEE float WAV requires 32-bit samples (got " + bitsPerSample + ")");
        }
        int expectedFrameBytes = channels * bytesPerSample;
        int frameStride = blockAlign >= expectedFrameBytes ? blockAlign : expectedFrameBytes;
        int frameCount = dataLength / frameStride;
        if (frameCount <= 0) {
            throw new IllegalArgumentException("WAV data chunk contains 0 complete audio frames");
        }
        checkDecodedSampleLimit(frameCount, maxMediaBytes);

        float[] mono = new float[frameCount];
        float invChannels = 1.0f / channels;
        for (int f = 0; f < frameCount; f++) {
            int framePos = dataOffset + f * frameStride;
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ch++) {
                int samplePos = framePos + ch * bytesPerSample;
                sum += decodeSample(buf, bytes, samplePos, audioFormat, bitsPerSample);
            }
            mono[f] = clampUnit(sum * invChannels);
        }

        if (sampleRate == WHISPER_SAMPLE_RATE_HZ) {
            return mono;
        }
        return resampleTo16kHz(mono, sampleRate, maxMediaBytes);
    }

    private static float decodeSample(ByteBuffer buf, byte[] raw, int pos, int audioFormat, int bitsPerSample) {
        if (audioFormat == WAVE_FORMAT_IEEE_FLOAT) {
            float v = buf.getFloat(pos);
            return Float.isFinite(v) ? v : 0.0f;
        }
        return switch (bitsPerSample) {
            case 8 -> ((raw[pos] & 0xFF) - 128) / 128.0f;
            case 16 -> buf.getShort(pos) / 32768.0f;
            case 24 -> {
                int b0 = raw[pos] & 0xFF;
                int b1 = raw[pos + 1] & 0xFF;
                int b2 = raw[pos + 2]; // signed MSB
                int s24 = b0 | (b1 << 8) | (b2 << 16);
                yield s24 / 8388608.0f;
            }
            case 32 -> buf.getInt(pos) / 2147483648.0f;
            default -> throw new IllegalArgumentException("Unsupported PCM bit depth: " + bitsPerSample);
        };
    }

    private static float[] resampleTo16kHz(float[] mono, int srcRate, long maxMediaBytes) {
        long rawTargetLen = Math.max(1L, Math.round((double) mono.length * WHISPER_SAMPLE_RATE_HZ / (double) srcRate));
        if (rawTargetLen > Integer.MAX_VALUE / 4) {
            throw new IllegalArgumentException("Resampled audio length overflow: " + rawTargetLen);
        }
        int outLen = (int) rawTargetLen;
        checkDecodedSampleLimit(outLen, maxMediaBytes);

        float[] out = new float[outLen];
        if (mono.length == 1) {
            out[0] = mono[0];
            return out;
        }
        double ratio = (double) srcRate / (double) WHISPER_SAMPLE_RATE_HZ;
        if (ratio > 1.0) {
            // Anti-aliased box-window averaging when downsampling (e.g., 44.1kHz / 48kHz -> 16kHz)
            for (int i = 0; i < outLen; i++) {
                double start = i * ratio;
                double end = Math.min(mono.length, (i + 1) * ratio);
                int iStart = Math.min(mono.length - 1, (int) Math.floor(start));
                int iEnd = Math.min(mono.length, Math.max(iStart + 1, (int) Math.ceil(end)));
                float acc = 0.0f;
                for (int j = iStart; j < iEnd; j++) {
                    acc += mono[j];
                }
                out[i] = acc / (iEnd - iStart);
            }
        } else {
            // Linear interpolation when upsampling (e.g., 8kHz -> 16kHz)
            double scale = (double) (mono.length - 1) / Math.max(1, outLen - 1);
            for (int i = 0; i < outLen; i++) {
                double srcIdx = i * scale;
                int idx0 = (int) Math.floor(srcIdx);
                int idx1 = Math.min(mono.length - 1, idx0 + 1);
                float frac = (float) (srcIdx - idx0);
                out[i] = mono[idx0] + frac * (mono[idx1] - mono[idx0]);
            }
        }
        return out;
    }

    private static void checkDecodedSampleLimit(int sampleCount, long maxMediaBytes) {
        long floatBytes = (long) sampleCount * Float.BYTES;
        long maxExpandedBytes = Math.max(maxMediaBytes * 4L, 16L * 1024L * 1024L);
        if (maxExpandedBytes < maxMediaBytes) {
            maxExpandedBytes = Long.MAX_VALUE;
        }
        if (floatBytes > maxExpandedBytes) {
            throw new IllegalArgumentException(
                "Decoded PCM buffer (" + floatBytes + " bytes) exceeds safe memory ceiling (" + maxExpandedBytes + " bytes)"
            );
        }
    }

    private static float clampUnit(float v) {
        if (v > 1.0f) return 1.0f;
        if (v < -1.0f) return -1.0f;
        return v;
    }
}
