// Performance test comparing all synthesis implementations
// Tests: uncached, optimized (GPU cache), graph-reuse (batched context + allocator reuse)

#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>

// Helper to write WAV file
static bool write_wav(const char * path, const std::vector<float> & audio, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;

    int32_t data_size = audio.size() * sizeof(int16_t);
    int32_t file_size = 36 + data_size;

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    int32_t fmt_size = 16;
    int16_t audio_format = 1;
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

    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

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
    const char * text_input = nullptr;
    const char * output_wav = "output_graph_reuse.wav";
    bool compare_all = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text_input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_wav = argv[++i];
        } else if (strcmp(argv[i], "--compare") == 0) {
            compare_all = true;
        }
    }

    printf("=== Magpie TTS Graph-Reuse Performance Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Codec: %s\n", codec_path);
    printf("Output: %s\n", output_wav);
    printf("\n");

    // Load model
    printf("Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("Backend: %s\n\n", magpie_get_backend_name(ctx));

    // Tokenize input
    std::vector<int32_t> tokens;
    if (text_input) {
        printf("Input: \"%s\"\n", text_input);
        tokens = magpie_tokenize(&ctx->model.tokenizer, text_input);
    } else {
        text_input = "Hello, this is a test of the Magpie text to speech system.";
        printf("Default: \"%s\"\n", text_input);
        tokens = magpie_tokenize(&ctx->model.tokenizer, text_input);
    }
    printf("Tokens: %zu\n\n", tokens.size());

    // Set inference params
    ctx->temperature = 0.7f;
    ctx->top_k = 80;
    ctx->speaker_id = 0;
    ctx->model.hparams.max_dec_steps = 500;

    // Run graph-reuse version
    printf("=== Running Graph-Reuse Version ===\n");
    auto start_gr = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> codes_gr = magpie_synthesize_codes_graph_reuse(ctx, tokens.data(), tokens.size());
    auto end_gr = std::chrono::high_resolution_clock::now();
    double time_gr = std::chrono::duration<double>(end_gr - start_gr).count();

    if (codes_gr.empty()) {
        fprintf(stderr, "Graph-reuse synthesis failed!\n");
        magpie_free(ctx);
        return 1;
    }

    int n_frames_gr = codes_gr.size() / 8;
    printf("\nGraph-Reuse: %d frames in %.3f sec (%.1f fps)\n\n",
           n_frames_gr, time_gr, n_frames_gr / time_gr);

    // Compare with other versions
    if (compare_all) {
        printf("=== Running Comparison ===\n\n");

        // GPU-optimized (old)
        printf("Running GPU-Optimized version...\n");
        auto start_opt = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> codes_opt = magpie_synthesize_codes_optimized(ctx, tokens.data(), tokens.size());
        auto end_opt = std::chrono::high_resolution_clock::now();
        double time_opt = std::chrono::duration<double>(end_opt - start_opt).count();
        int n_frames_opt = codes_opt.size() / 8;

        // Uncached
        printf("Running Uncached version...\n");
        auto start_unc = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> codes_unc = magpie_synthesize_codes(ctx, tokens.data(), tokens.size());
        auto end_unc = std::chrono::high_resolution_clock::now();
        double time_unc = std::chrono::duration<double>(end_unc - start_unc).count();
        int n_frames_unc = codes_unc.size() / 8;

        printf("\n");
        printf("=== Performance Comparison ===\n");
        printf("+-----------------+--------+----------+----------+\n");
        printf("| Version         | Frames | Time (s) | FPS      |\n");
        printf("+-----------------+--------+----------+----------+\n");
        printf("| Uncached        | %6d | %8.3f | %8.1f |\n",
               n_frames_unc, time_unc, n_frames_unc / time_unc);
        printf("| GPU-Optimized   | %6d | %8.3f | %8.1f |\n",
               n_frames_opt, time_opt, n_frames_opt / time_opt);
        printf("| Graph-Reuse     | %6d | %8.3f | %8.1f |\n",
               n_frames_gr, time_gr, n_frames_gr / time_gr);
        printf("+-----------------+--------+----------+----------+\n");
        printf("\n");
        printf("Speedup vs Uncached:\n");
        printf("  GPU-Optimized: %.2fx\n", time_unc / time_opt);
        printf("  Graph-Reuse:   %.2fx\n", time_unc / time_gr);
        printf("\n");
        printf("Speedup vs GPU-Optimized:\n");
        printf("  Graph-Reuse:   %.2fx\n", time_opt / time_gr);
    }

    // Decode to audio
    printf("\n=== Decoding to Audio ===\n");
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "Failed to load codec\n");
        magpie_free(ctx);
        return 1;
    }

    const int CHUNK_SIZE = 32;
    std::vector<float> audio;
    int n_frames = codes_gr.size() / 8;

    for (int chunk_start = 0; chunk_start < n_frames; chunk_start += CHUNK_SIZE) {
        int chunk_frames = std::min(CHUNK_SIZE, n_frames - chunk_start);

        std::vector<int32_t> chunk_codes(chunk_frames * 8);
        for (int t = 0; t < chunk_frames; t++) {
            for (int cb = 0; cb < 8; cb++) {
                chunk_codes[cb * chunk_frames + t] = codes_gr[(chunk_start + t) * 8 + cb];
            }
        }

        std::vector<float> chunk_audio = magpie_codec_decode(codec, chunk_codes.data(), chunk_frames);
        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
    }

    printf("Decoded %zu samples (%.2f sec @ 22050 Hz)\n",
           audio.size(), (float)audio.size() / 22050.0f);

    if (write_wav(output_wav, audio, 22050)) {
        printf("Saved: %s\n", output_wav);
    }

    magpie_codec_free(codec);
    magpie_free(ctx);

    printf("\n=== Done ===\n");
    return 0;
}
