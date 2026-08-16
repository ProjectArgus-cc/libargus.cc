package cc.projectargus.libargus;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Utility helper to serialize raw float PCM samples into standard 16-bit WAVE files
 * using the built-in JDK javax.sound.sampled library.
 */
final class WavWriter {

    private WavWriter() {
        // Prevent instantiation
    }

    /**
     * Writes raw float PCM samples to a .wav file.
     *
     * @param pcm          raw float samples in the range [-1.0, 1.0]
     * @param sampleRate   sample rate in Hz (e.g. 16000 or 24000)
     * @param outputFile   destination file path
     * @throws IOException if writing fails
     */
    static void writeWav(float[] pcm, float sampleRate, File outputFile) throws IOException {
        byte[] byteData = new byte[pcm.length * 2];
        for (int i = 0; i < pcm.length; i++) {
            // Clamp float sample to prevent clipping/overflow
            float sample = Math.max(-1.0f, Math.min(1.0f, pcm[i]));
            short value = (short) (sample * 32767.0f);
            byteData[i * 2] = (byte) (value & 0xff);
            byteData[i * 2 + 1] = (byte) ((value >> 8) & 0xff);
        }

        AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false); // 16-bit, mono, signed, little-endian
        try (ByteArrayInputStream bais = new ByteArrayInputStream(byteData);
             AudioInputStream ais = new AudioInputStream(bais, format, pcm.length)) {
            
            // Ensure parent directories exist
            File parent = outputFile.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }

            AudioSystem.write(ais, AudioFileFormat.Type.WAVE, outputFile);
        }
    }

    /**
     * Writes raw float PCM samples from an unmanaged Panama MemorySegment to a .wav file.
     *
     * @param pcmSegment   off-heap float PCM samples segment
     * @param sampleCount  total number of float samples to read
     * @param sampleRate   sample rate in Hz
     * @param outputFile   destination file path
     * @throws IOException if writing fails
     */
    static void writeWav(MemorySegment pcmSegment, int sampleCount, float sampleRate, File outputFile) throws IOException {
        float[] pcm = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            pcm[i] = pcmSegment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        writeWav(pcm, sampleRate, outputFile);
    }
}
