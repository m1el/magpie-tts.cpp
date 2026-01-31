// End-to-end inference test for Magpie TTS using KV-cached synthesis
// Tests the optimized pipeline with O(n) per-step complexity

#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>

// Helper to load tokens from a binary file
static std::vector<int32_t> load_tokens(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};

    // Read shape header (4 x int64)
    int64_t shape[4];
    f.read(reinterpret_cast<char*>(shape), 4 * sizeof(int64_t));

    // Find the non-1 dimension
    int64_t n_tokens = 1;
    for (int i = 0; i < 4; i++) {
        if (shape[i] > 1) n_tokens = shape[i];
    }

    // Read float data (stored as float32 from Python)
    std::vector<float> float_data(n_tokens);
    f.read(reinterpret_cast<char*>(float_data.data()), n_tokens * sizeof(float));

    // Convert to int32
    std::vector<int32_t> tokens(n_tokens);
    for (int64_t i = 0; i < n_tokens; i++) {
        tokens[i] = static_cast<int32_t>(float_data[i]);
    }

    return tokens;
}

// Helper to write WAV file
static bool write_wav(const char * path, const std::vector<float> & audio, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;

    int32_t data_size = audio.size() * sizeof(int16_t);
    int32_t file_size = 36 + data_size;

    // WAV header
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

    // data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    // Convert float to int16 and write
    for (float sample : audio) {
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t s = static_cast<int16_t>(sample * 32767.0f);
        fwrite(&s, 2, 1, f);
    }

    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * codec_path = "weights/nano-codec-f32.gguf";
    const char * tokens_path = "test_data/reference/manual_text_tokens.bin";
    const char * output_wav = "output_cached.wav";
    bool compare_with_uncached = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_wav = argv[++i];
        } else if (strcmp(argv[i], "--compare") == 0) {
            compare_with_uncached = true;
        }
    }

    printf("=== Magpie TTS KV-Cached Inference Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Codec: %s\n", codec_path);
    printf("Tokens: %s\n", tokens_path);
    printf("Output: %s\n", output_wav);
    printf("\n");

    // Load tokens
    std::vector<int32_t> tokens = load_tokens(tokens_path);
    if (tokens.empty()) {
        fprintf(stderr, "Failed to load tokens from %s\n", tokens_path);
        fprintf(stderr, "Run: uv run scripts/dump_reference.py --text \"Hello world\"\n");
        return 1;
    }
    printf("Loaded %zu tokens\n", tokens.size());

    // Initialize model
    printf("Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }
    printf("Model loaded, backend: %s\n", magpie_get_backend_name(ctx));

    // Set inference parameters
    ctx->temperature = 0.7f;
    ctx->top_k = 80;
    ctx->speaker_id = 0;
    ctx->model.hparams.max_dec_steps = 50;  // Limit for testing

    // Synthesize with cached version
    printf("\n=== Synthesizing Audio Codes (KV-Cached) ===\n");
    auto start_cached = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> codes = magpie_synthesize_codes_cached(ctx, tokens.data(), tokens.size());
    auto end_cached = std::chrono::high_resolution_clock::now();
    double cached_time = std::chrono::duration<double>(end_cached - start_cached).count();

    if (codes.empty()) {
        fprintf(stderr, "Cached synthesis failed!\n");
        magpie_free(ctx);
        return 1;
    }

    int n_frames = codes.size() / 8;
    printf("Generated %d audio frames (%zu codes) in %.2f seconds\n",
           n_frames, codes.size(), cached_time);
    printf("Speed: %.1f frames/second\n", n_frames / cached_time);

    // Print first few frames
    printf("First 3 frames:\n");
    for (int t = 0; t < std::min(3, n_frames); t++) {
        printf("  Frame %d:", t);
        for (int cb = 0; cb < 8; cb++) {
            printf(" %d", codes[t * 8 + cb]);
        }
        printf("\n");
    }

    // Optionally compare with uncached version
    if (compare_with_uncached) {
        printf("\n=== Synthesizing Audio Codes (Uncached - for comparison) ===\n");
        auto start_uncached = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> codes_uncached = magpie_synthesize_codes(ctx, tokens.data(), tokens.size());
        auto end_uncached = std::chrono::high_resolution_clock::now();
        double uncached_time = std::chrono::duration<double>(end_uncached - start_uncached).count();

        printf("Uncached: %.2f seconds, Cached: %.2f seconds\n", uncached_time, cached_time);
        printf("Speedup: %.2fx\n", uncached_time / cached_time);

        // Compare codes
        if (codes_uncached.size() == codes.size()) {
            int diff_count = 0;
            for (size_t i = 0; i < codes.size(); i++) {
                if (codes[i] != codes_uncached[i]) diff_count++;
            }
            printf("Code differences: %d / %zu (%.1f%%)\n",
                   diff_count, codes.size(), 100.0 * diff_count / codes.size());
        }
    }

    // Load codec and decode to audio
    printf("\n=== Decoding to Audio ===\n");
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "Failed to load codec from %s\n", codec_path);
        magpie_free(ctx);
        return 1;
    }

    // Reorder codes from [n_frames * 8] flat to [8, n_frames] for codec
    std::vector<int32_t> codes_reordered(codes.size());
    for (int t = 0; t < n_frames; t++) {
        for (int cb = 0; cb < 8; cb++) {
            codes_reordered[cb * n_frames + t] = codes[t * 8 + cb];
        }
    }

    std::vector<float> audio = magpie_codec_decode(codec, codes_reordered.data(), n_frames);
    if (audio.empty()) {
        fprintf(stderr, "Codec decode failed!\n");
        magpie_codec_free(codec);
        magpie_free(ctx);
        return 1;
    }

    printf("Decoded %zu audio samples (%.2f seconds at %d Hz)\n",
           audio.size(), (float)audio.size() / 22050.0f, 22050);

    // Write WAV
    if (write_wav(output_wav, audio, 22050)) {
        printf("Saved audio to %s\n", output_wav);
    } else {
        fprintf(stderr, "Failed to write WAV file\n");
    }

    // Cleanup
    magpie_codec_free(codec);
    magpie_free(ctx);

    printf("\n=== Test Complete ===\n");
    return 0;
}
