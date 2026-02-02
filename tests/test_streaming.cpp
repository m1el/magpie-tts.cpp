// Streaming TTS test - generates and outputs audio incrementally
// No hard limit on frames, relies on EOS detection

#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>

// Helper to write WAV header (can update later for streaming)
static void write_wav_header(FILE * f, int sample_rate, int data_size) {
    int32_t file_size = 36 + data_size;

    fseek(f, 0, SEEK_SET);
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    int32_t fmt_size = 16;
    int16_t audio_format = 1;  // PCM
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * 2;
    int16_t block_align = 2;
    int16_t bits_per_sample = 16;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    // data chunk header
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
}

// Write audio samples to file
static void write_audio_samples(FILE * f, const std::vector<float> & audio) {
    for (float sample : audio) {
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t s = static_cast<int16_t>(sample * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * codec_path = "weights/nano-codec-f32.gguf";
    const char * text_input = nullptr;
    const char * output_wav = "output_streaming.wav";
    int chunk_frames = 20;  // Decode every N frames

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text_input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_wav = argv[++i];
        } else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            chunk_frames = atoi(argv[++i]);
        }
    }

    printf("=== Magpie TTS Streaming Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Codec: %s\n", codec_path);
    printf("Output: %s\n", output_wav);
    printf("Chunk size: %d frames\n", chunk_frames);
    printf("\n");

    // Initialize model
    printf("Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Model loaded, backend: %s\n", magpie_get_backend_name(ctx));

    // Get text input
    if (!text_input) {
        text_input = "Hello, this is a streaming test of the Magpie text to speech system. "
                     "It should generate audio incrementally without any hard limits.";
    }
    printf("Input: \"%s\"\n", text_input);

    // Tokenize
    std::vector<int32_t> tokens = magpie_tokenize(&ctx->model.tokenizer, text_input);
    if (tokens.empty()) {
        fprintf(stderr, "Tokenization failed\n");
        magpie_free(ctx);
        return 1;
    }
    printf("Tokenized to %zu tokens\n\n", tokens.size());

    // Set parameters - no practical limit
    ctx->temperature = 0.7f;
    ctx->top_k = 80;
    ctx->speaker_id = 0;
    ctx->model.hparams.max_dec_steps = 10000;  // Very high limit, rely on EOS

    // Load codec
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "Failed to load codec\n");
        magpie_free(ctx);
        return 1;
    }

    // Open output file
    FILE * wav_file = fopen(output_wav, "wb");
    if (!wav_file) {
        fprintf(stderr, "Failed to open output file\n");
        magpie_codec_free(codec);
        magpie_free(ctx);
        return 1;
    }

    // Write placeholder header (will update at end)
    write_wav_header(wav_file, 22050, 0);

    printf("=== Generating Audio (Streaming) ===\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    // Generate codes using optimized version
    std::vector<int32_t> all_codes = magpie_synthesize_codes_optimized(ctx, tokens.data(), tokens.size());

    auto gen_time = std::chrono::high_resolution_clock::now();
    double gen_seconds = std::chrono::duration<double>(gen_time - start_time).count();

    if (all_codes.empty()) {
        fprintf(stderr, "Synthesis failed\n");
        fclose(wav_file);
        magpie_codec_free(codec);
        magpie_free(ctx);
        return 1;
    }

    int total_frames = all_codes.size() / 8;
    printf("Generated %d frames in %.2f seconds (%.1f fps)\n",
           total_frames, gen_seconds, total_frames / gen_seconds);

    // Decode in chunks
    printf("\n=== Decoding Audio (Chunked) ===\n");
    int total_samples = 0;
    int frames_decoded = 0;

    while (frames_decoded < total_frames) {
        int chunk_size = std::min(chunk_frames, total_frames - frames_decoded);

        // Reorder codes for this chunk: [n_frames * 8] -> [8, n_frames]
        std::vector<int32_t> chunk_codes(chunk_size * 8);
        for (int t = 0; t < chunk_size; t++) {
            for (int cb = 0; cb < 8; cb++) {
                chunk_codes[cb * chunk_size + t] = all_codes[(frames_decoded + t) * 8 + cb];
            }
        }

        // Decode chunk
        std::vector<float> audio = magpie_codec_decode(codec, chunk_codes.data(), chunk_size);

        if (audio.empty()) {
            fprintf(stderr, "Decode failed at frame %d\n", frames_decoded);
            break;
        }

        // Write to file
        write_audio_samples(wav_file, audio);
        total_samples += audio.size();
        frames_decoded += chunk_size;

        printf("  Decoded frames %d-%d (%d samples, %.2f sec cumulative)\n",
               frames_decoded - chunk_size, frames_decoded - 1,
               (int)audio.size(), (float)total_samples / 22050.0f);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    // Update WAV header with final size
    int data_size = total_samples * sizeof(int16_t);
    write_wav_header(wav_file, 22050, data_size);
    fclose(wav_file);

    float audio_duration = (float)total_samples / 22050.0f;
    printf("\n=== Summary ===\n");
    printf("Total frames: %d\n", total_frames);
    printf("Audio duration: %.2f seconds\n", audio_duration);
    printf("Generation time: %.2f seconds\n", gen_seconds);
    printf("Total time (gen + decode): %.2f seconds\n", total_seconds);
    printf("Real-time factor: %.2fx\n", audio_duration / total_seconds);
    printf("Saved to: %s\n", output_wav);

    // Cleanup
    magpie_codec_free(codec);
    magpie_free(ctx);

    return 0;
}
