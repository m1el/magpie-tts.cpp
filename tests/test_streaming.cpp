// Streaming TTS test - demonstrates real-time audio output with callbacks
// Audio is generated and delivered via callbacks as it becomes available

#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>

// Streaming state
struct stream_state {
    std::vector<float> all_audio;
    int chunks_received;
    int total_samples;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point first_audio_time;
    bool first_chunk;
    FILE * raw_output;  // Optional: write raw PCM as it arrives
};

// Audio callback - called for each chunk of audio
bool on_audio_chunk(const float * samples, int n_samples, void * user_data) {
    stream_state * state = (stream_state *)user_data;

    if (state->first_chunk) {
        state->first_audio_time = std::chrono::high_resolution_clock::now();
        state->first_chunk = false;
        double latency_ms = std::chrono::duration<double, std::milli>(
            state->first_audio_time - state->start_time).count();
        fprintf(stderr, "[STREAM] First audio chunk! Latency: %.1f ms\n", latency_ms);
    }

    // Store audio
    state->all_audio.insert(state->all_audio.end(), samples, samples + n_samples);
    state->chunks_received++;
    state->total_samples += n_samples;

    // Write to raw output if enabled
    if (state->raw_output) {
        std::vector<int16_t> pcm(n_samples);
        for (int i = 0; i < n_samples; i++) {
            float s = samples[i];
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            pcm[i] = (int16_t)(s * 32767.0f);
        }
        fwrite(pcm.data(), sizeof(int16_t), n_samples, state->raw_output);
        fflush(state->raw_output);
    }

    double audio_ms = (double)n_samples / 22.05;  // 22050 Hz
    fprintf(stderr, "[STREAM] Chunk %d: %d samples (%.1f ms audio)\n",
            state->chunks_received, n_samples, audio_ms);

    return true;  // Continue generation
}

// Progress callback
void on_progress(int frames, int sentence_idx, int total_sentences, void * user_data) {
    (void)user_data;
    fprintf(stderr, "[PROGRESS] Sentence %d/%d, %d frames generated\n",
            sentence_idx + 1, total_sentences, frames);
}

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
        int16_t s = (int16_t)(sample * 32767.0f);
        fwrite(&s, 2, 1, f);
    }

    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * codec_path = "weights/nano-codec-f32.gguf";
    const char * text = nullptr;
    const char * output_path = "output_streaming.wav";
    const char * raw_output_path = nullptr;
    int frames_per_chunk = 4;
    bool sentence_chunking = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--raw") == 0 && i + 1 < argc) {
            raw_output_path = argv[++i];
        } else if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            frames_per_chunk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-sentence-chunk") == 0) {
            sentence_chunking = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Streaming TTS Test\n\n");
            printf("Usage: %s [options]\n\n", argv[0]);
            printf("Options:\n");
            printf("  --model PATH        Model GGUF path\n");
            printf("  --codec PATH        Codec GGUF path\n");
            printf("  --text TEXT         Text to synthesize\n");
            printf("  --output PATH       Output WAV file\n");
            printf("  --raw PATH          Write raw PCM as it streams (for piping)\n");
            printf("  --chunk-size N      Frames per audio chunk (default: 4)\n");
            printf("  --no-sentence-chunk Disable sentence-level chunking\n");
            return 0;
        }
    }

    if (!text) {
        text = "Hello! This is a streaming test. Each sentence is processed separately. "
               "Audio chunks are output as they become available.";
    }

    printf("=== Streaming TTS Test ===\n");
    printf("Text: \"%s\"\n", text);
    printf("Frames per chunk: %d (~%.0f ms audio latency)\n",
           frames_per_chunk, frames_per_chunk * 1024.0 / 22.05);
    printf("Sentence chunking: %s\n\n", sentence_chunking ? "enabled" : "disabled");

    // Load model
    printf("Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Load codec
    printf("Loading codec...\n");
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "Failed to load codec\n");
        magpie_free(ctx);
        return 1;
    }

    // Initialize streaming state
    stream_state state = {};
    state.first_chunk = true;
    state.start_time = std::chrono::high_resolution_clock::now();

    if (raw_output_path) {
        state.raw_output = fopen(raw_output_path, "wb");
        if (!state.raw_output) {
            fprintf(stderr, "Warning: Could not open raw output file\n");
        }
    }

    // Set up streaming parameters
    magpie_stream_params params;
    params.temperature = 0.7f;
    params.top_k = 80;
    params.speaker_id = 0;
    params.frames_per_chunk = frames_per_chunk;
    params.sentence_chunking = sentence_chunking;
    params.on_audio = on_audio_chunk;
    params.on_progress = on_progress;
    params.user_data = &state;

    // Run streaming synthesis
    printf("\n=== Starting Streaming Synthesis ===\n");
    int total_samples = magpie_synthesize_streaming(ctx, codec, text, params);

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - state.start_time).count();

    if (state.raw_output) {
        fclose(state.raw_output);
    }

    if (total_samples < 0) {
        fprintf(stderr, "Streaming synthesis failed!\n");
        magpie_codec_free(codec);
        magpie_free(ctx);
        return 1;
    }

    // Results
    printf("\n=== Streaming Complete ===\n");
    printf("Total chunks: %d\n", state.chunks_received);
    printf("Total samples: %d (%.2f seconds audio)\n",
           total_samples, total_samples / 22050.0f);
    printf("Total time: %.2f seconds\n", total_time);
    printf("Real-time factor: %.2fx\n", (total_samples / 22050.0f) / total_time);

    if (!state.first_chunk) {
        double latency = std::chrono::duration<double, std::milli>(
            state.first_audio_time - state.start_time).count();
        printf("Time to first audio: %.1f ms\n", latency);
    }

    // Save final WAV
    if (!state.all_audio.empty()) {
        if (write_wav(output_path, state.all_audio, 22050)) {
            printf("Saved: %s\n", output_path);
        }
    }

    magpie_codec_free(codec);
    magpie_free(ctx);

    return 0;
}
