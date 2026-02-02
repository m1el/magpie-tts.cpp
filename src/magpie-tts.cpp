// Magpie TTS - Text-to-Speech using GGML
// Usage: magpie-tts [options] --text "Your text here"

#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr, "Magpie TTS - Text-to-Speech using GGML\n\n");
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model PATH     Path to model GGUF (default: weights/magpie-357m-f32.gguf)\n");
    fprintf(stderr, "  -c, --codec PATH     Path to codec GGUF (default: weights/nano-codec-f32.gguf)\n");
    fprintf(stderr, "  -t, --text TEXT      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output PATH    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -s, --speaker ID     Speaker ID (default: 0)\n");
    fprintf(stderr, "  --temp FLOAT         Sampling temperature (default: 0.7, 0=deterministic)\n");
    fprintf(stderr, "  --top-k INT          Top-k sampling (default: 80)\n");
    fprintf(stderr, "  -q, --quiet          Minimal output\n");
    fprintf(stderr, "  -h, --help           Show this help\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s -t \"Hello, world!\"\n", prog);
    fprintf(stderr, "  %s -t \"Hello\" -o hello.wav --temp 0.5\n", prog);
    fprintf(stderr, "  %s -m model.gguf -c codec.gguf -t \"Test\"\n", prog);
}

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
    // Default parameters
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * codec_path = "weights/nano-codec-f32.gguf";
    const char * text_input = nullptr;
    const char * output_path = "output.wav";
    int speaker_id = 0;
    float temperature = 0.7f;
    int top_k = 80;
    bool quiet = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "Error: --model requires a path\n"); return 1; }
            model_path = argv[i];
        } else if (arg == "-c" || arg == "--codec") {
            if (++i >= argc) { fprintf(stderr, "Error: --codec requires a path\n"); return 1; }
            codec_path = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) { fprintf(stderr, "Error: --text requires text\n"); return 1; }
            text_input = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { fprintf(stderr, "Error: --output requires a path\n"); return 1; }
            output_path = argv[i];
        } else if (arg == "-s" || arg == "--speaker") {
            if (++i >= argc) { fprintf(stderr, "Error: --speaker requires an ID\n"); return 1; }
            speaker_id = atoi(argv[i]);
        } else if (arg == "--temp") {
            if (++i >= argc) { fprintf(stderr, "Error: --temp requires a value\n"); return 1; }
            temperature = atof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) { fprintf(stderr, "Error: --top-k requires a value\n"); return 1; }
            top_k = atoi(argv[i]);
        } else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required args
    if (!text_input) {
        fprintf(stderr, "Error: --text is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!quiet) {
        fprintf(stderr, "Magpie TTS\n");
        fprintf(stderr, "  Model: %s\n", model_path);
        fprintf(stderr, "  Codec: %s\n", codec_path);
        fprintf(stderr, "  Text: \"%s\"\n", text_input);
        fprintf(stderr, "  Output: %s\n", output_path);
        fprintf(stderr, "  Speaker: %d\n", speaker_id);
        fprintf(stderr, "  Temperature: %.2f\n", temperature);
        fprintf(stderr, "  Top-k: %d\n\n", top_k);
    }

    // Load model
    if (!quiet) fprintf(stderr, "Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_path);
        return 1;
    }
    if (!quiet) fprintf(stderr, "Backend: %s\n\n", magpie_get_backend_name(ctx));

    // Set parameters
    ctx->temperature = temperature;
    ctx->top_k = top_k;
    ctx->speaker_id = speaker_id;

    // Tokenize
    if (!quiet) fprintf(stderr, "Tokenizing...\n");
    std::vector<int32_t> tokens = magpie_tokenize(&ctx->model.tokenizer, text_input);
    if (tokens.empty()) {
        fprintf(stderr, "Error: Tokenization failed\n");
        magpie_free(ctx);
        return 1;
    }
    if (!quiet) fprintf(stderr, "Tokens: %zu\n\n", tokens.size());

    // Synthesize using graph-reuse (fastest)
    if (!quiet) fprintf(stderr, "Synthesizing...\n");
    std::vector<int32_t> codes = magpie_synthesize_codes_graph_reuse(ctx, tokens.data(), tokens.size());
    if (codes.empty()) {
        fprintf(stderr, "Error: Synthesis failed\n");
        magpie_free(ctx);
        return 1;
    }

    int n_frames = codes.size() / 8;
    if (!quiet) fprintf(stderr, "Generated %d frames\n\n", n_frames);

    // Load codec
    if (!quiet) fprintf(stderr, "Loading codec...\n");
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "Error: Failed to load codec from %s\n", codec_path);
        magpie_free(ctx);
        return 1;
    }

    // Decode to audio (in chunks for CUDA compatibility)
    if (!quiet) fprintf(stderr, "Decoding audio...\n");
    const int CHUNK_SIZE = 32;
    std::vector<float> audio;

    for (int chunk_start = 0; chunk_start < n_frames; chunk_start += CHUNK_SIZE) {
        int chunk_frames = std::min(CHUNK_SIZE, n_frames - chunk_start);

        // Reorder codes: [frames, 8] -> [8, frames]
        std::vector<int32_t> chunk_codes(chunk_frames * 8);
        for (int t = 0; t < chunk_frames; t++) {
            for (int cb = 0; cb < 8; cb++) {
                chunk_codes[cb * chunk_frames + t] = codes[(chunk_start + t) * 8 + cb];
            }
        }

        std::vector<float> chunk_audio = magpie_codec_decode(codec, chunk_codes.data(), chunk_frames);
        if (chunk_audio.empty()) {
            fprintf(stderr, "Error: Codec decode failed\n");
            magpie_codec_free(codec);
            magpie_free(ctx);
            return 1;
        }

        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
    }

    // Write WAV
    if (!quiet) fprintf(stderr, "Writing %s...\n", output_path);
    if (!write_wav(output_path, audio, 22050)) {
        fprintf(stderr, "Error: Failed to write %s\n", output_path);
        magpie_codec_free(codec);
        magpie_free(ctx);
        return 1;
    }

    float duration = (float)audio.size() / 22050.0f;
    if (!quiet) {
        fprintf(stderr, "\nDone! Generated %.2f seconds of audio.\n", duration);
    } else {
        printf("%s\n", output_path);
    }

    // Cleanup
    magpie_codec_free(codec);
    magpie_free(ctx);

    return 0;
}
