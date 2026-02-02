// End-to-end inference test for Magpie TTS using GPU-optimized KV cache
// Tests the fully optimized pipeline with GPU-resident cache (no CPU round-trips)

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
    const char * tokens_path = nullptr;
    const char * text_input = nullptr;
    const char * output_wav = "output_optimized.wav";
    bool compare_all = false;
    bool deterministic = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text_input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_wav = argv[++i];
        } else if (strcmp(argv[i], "--compare-all") == 0) {
            compare_all = true;
        } else if (strcmp(argv[i], "--deterministic") == 0) {
            deterministic = true;
        }
    }

    printf("=== Magpie TTS GPU-Optimized KV Cache Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Codec: %s\n", codec_path);
    printf("Output: %s\n", output_wav);
    printf("\n");

    // Initialize model first (needed for tokenizer)
    printf("Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }
    printf("Model loaded, backend: %s\n", magpie_get_backend_name(ctx));

    // Get tokens: either from text or from file
    std::vector<int32_t> tokens;

    if (text_input) {
        // Tokenize text using built-in tokenizer
        printf("Input text: \"%s\"\n", text_input);
        tokens = magpie_tokenize(&ctx->model.tokenizer, text_input);
        if (tokens.empty()) {
            fprintf(stderr, "Tokenization failed!\n");
            magpie_free(ctx);
            return 1;
        }
        printf("Tokenized to %zu tokens\n", tokens.size());
    } else if (tokens_path) {
        // Load pre-tokenized tokens from file
        printf("Tokens file: %s\n", tokens_path);
        tokens = load_tokens(tokens_path);
        if (tokens.empty()) {
            fprintf(stderr, "Failed to load tokens from %s\n", tokens_path);
            magpie_free(ctx);
            return 1;
        }
        printf("Loaded %zu tokens\n", tokens.size());
    } else {
        // Default: use built-in test text
        text_input = "Hello, this is a test of the Magpie text to speech system.";
        printf("Using default text: \"%s\"\n", text_input);
        tokens = magpie_tokenize(&ctx->model.tokenizer, text_input);
        if (tokens.empty()) {
            fprintf(stderr, "Tokenization failed!\n");
            magpie_free(ctx);
            return 1;
        }
        printf("Tokenized to %zu tokens\n", tokens.size());
    }

    // Set inference parameters
    if (deterministic) {
        ctx->temperature = 0.0f;  // Argmax sampling
        ctx->top_k = 1;
        printf("Mode: Deterministic (argmax)\n");
    } else {
        ctx->temperature = 0.7f;
        ctx->top_k = 80;
        printf("Mode: Stochastic (temp=0.7, top_k=80)\n");
    }
    ctx->speaker_id = 0;
    ctx->model.hparams.max_dec_steps = 500;  // Allow natural EOS detection

    // Synthesize with GPU-optimized version
    printf("\n=== Synthesizing Audio Codes (GPU-Optimized) ===\n");
    auto start_opt = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> codes = magpie_synthesize_codes_optimized(ctx, tokens.data(), tokens.size());
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_time = std::chrono::duration<double>(end_opt - start_opt).count();

    if (codes.empty()) {
        fprintf(stderr, "GPU-optimized synthesis failed!\n");
        magpie_free(ctx);
        return 1;
    }

    int n_frames = codes.size() / 8;
    printf("Generated %d audio frames (%zu codes) in %.2f seconds\n",
           n_frames, codes.size(), opt_time);
    printf("Speed: %.1f frames/second\n", n_frames / opt_time);

    // Print first few frames
    printf("First 3 frames:\n");
    for (int t = 0; t < std::min(3, n_frames); t++) {
        printf("  Frame %d:", t);
        for (int cb = 0; cb < 8; cb++) {
            printf(" %d", codes[t * 8 + cb]);
        }
        printf("\n");
    }

    // Compare with other versions
    if (compare_all) {
        printf("\n=== Comparison with Other Versions ===\n");

        // Uncached version
        printf("\nRunning uncached version...\n");
        auto start_uncached = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> codes_uncached = magpie_synthesize_codes(ctx, tokens.data(), tokens.size());
        auto end_uncached = std::chrono::high_resolution_clock::now();
        double uncached_time = std::chrono::duration<double>(end_uncached - start_uncached).count();

        // Cached version
        printf("Running cached version...\n");
        auto start_cached = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> codes_cached = magpie_synthesize_codes_cached(ctx, tokens.data(), tokens.size());
        auto end_cached = std::chrono::high_resolution_clock::now();
        double cached_time = std::chrono::duration<double>(end_cached - start_cached).count();

        printf("\nTiming Results:\n");
        printf("  Uncached:      %.2f seconds\n", uncached_time);
        printf("  Cached:        %.2f seconds\n", cached_time);
        printf("  GPU-Optimized: %.2f seconds\n", opt_time);
        printf("\nSpeedups vs Uncached:\n");
        printf("  Cached:        %.2fx\n", uncached_time / cached_time);
        printf("  GPU-Optimized: %.2fx\n", uncached_time / opt_time);

        // Compare outputs
        printf("\nOutput Comparison:\n");

        auto compare_codes = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b, const char* name) {
            if (a.size() != b.size()) {
                printf("  %s: size mismatch (%zu vs %zu)\n", name, a.size(), b.size());
                return;
            }
            int diff_count = 0;
            for (size_t i = 0; i < a.size(); i++) {
                if (a[i] != b[i]) diff_count++;
            }
            printf("  %s: %d / %zu codes differ (%.1f%%)\n",
                   name, diff_count, a.size(), 100.0 * diff_count / a.size());
        };

        compare_codes(codes, codes_uncached, "Optimized vs Uncached");
        compare_codes(codes, codes_cached, "Optimized vs Cached");
        compare_codes(codes_cached, codes_uncached, "Cached vs Uncached");
    }

    // Load codec and decode to audio
    printf("\n=== Decoding to Audio ===\n");
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "Failed to load codec from %s\n", codec_path);
        magpie_free(ctx);
        return 1;
    }

    // Decode in chunks to work around CUDA IM2COL issue with large frame counts
    const int CHUNK_SIZE = 32;  // Frames per chunk (32 works reliably)
    std::vector<float> audio;

    for (int chunk_start = 0; chunk_start < n_frames; chunk_start += CHUNK_SIZE) {
        int chunk_frames = std::min(CHUNK_SIZE, n_frames - chunk_start);

        // Reorder codes for this chunk: [chunk_frames * 8] -> [8, chunk_frames]
        std::vector<int32_t> chunk_codes(chunk_frames * 8);
        for (int t = 0; t < chunk_frames; t++) {
            for (int cb = 0; cb < 8; cb++) {
                chunk_codes[cb * chunk_frames + t] = codes[(chunk_start + t) * 8 + cb];
            }
        }

        std::vector<float> chunk_audio = magpie_codec_decode(codec, chunk_codes.data(), chunk_frames);
        if (chunk_audio.empty()) {
            fprintf(stderr, "Codec decode failed at chunk %d!\n", chunk_start / CHUNK_SIZE);
            magpie_codec_free(codec);
            magpie_free(ctx);
            return 1;
        }

        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
    }

    if (audio.empty()) {
        fprintf(stderr, "No audio decoded!\n");
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
