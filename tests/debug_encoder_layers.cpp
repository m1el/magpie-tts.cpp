#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>

// Read binary reference file (format: 4 x int64 shape + float32 data)
static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    if (fread(shape, sizeof(int64_t), 4, f) != 4) {
        fprintf(stderr, "Failed to read shape from: %s\n", path);
        fclose(f);
        return false;
    }

    int64_t n_elements = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n_elements);

    if (fread(data.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

static void compare_tensors(const char * name, const float * a, const float * b, size_t n) {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    size_t max_idx = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(a[i] - b[i]);
        sum_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    float mean_diff = sum_diff / n;
    fprintf(stderr, "  %s: max_diff=%.6f (at %zu), mean_diff=%.6f\n",
            name, max_diff, max_idx, mean_diff);

    if (max_diff > 0.01f) {
        fprintf(stderr, "    got[%zu]=%.6f, expected=%.6f\n", max_idx, a[max_idx], b[max_idx]);
    }
}

// Create causal mask
static void fill_causal_mask_f16(ggml_fp16_t * mask, int seq_len) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            mask[i * seq_len + j] = (j <= i) ? zero : neg_inf;
        }
    }
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    if (argc > 1) model_path = argv[1];

    // Load model
    fprintf(stderr, "Loading model from: %s\n", model_path);
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    magpie_encoder * encoder = &ctx->model.encoder;
    const int d_model = ctx->model.hparams.d_model;

    // Read text embedding input
    std::vector<float> text_emb;
    int64_t text_emb_shape[4];
    if (!read_reference("test_data/reference/manual_text_embedded.bin", text_emb, text_emb_shape)) {
        magpie_free(ctx);
        return 1;
    }
    const int seq_len = (int)text_emb_shape[2];
    fprintf(stderr, "Input: d_model=%d, seq_len=%d\n", d_model, seq_len);

    // Read expected outputs for each stage
    std::vector<float> enc_with_pos;
    int64_t enc_with_pos_shape[4];
    read_reference("test_data/reference/manual_enc_with_pos.bin", enc_with_pos, enc_with_pos_shape);

    std::vector<float> layer_expected[6];
    int64_t layer_shape[6][4];
    char path[256];
    for (int l = 0; l < 6; l++) {
        snprintf(path, sizeof(path), "test_data/reference/manual_enc_layer%d_out.bin", l);
        read_reference(path, layer_expected[l], layer_shape[l]);
    }

    std::vector<float> enc_output;
    int64_t enc_output_shape[4];
    read_reference("test_data/reference/manual_enc_output.bin", enc_output, enc_output_shape);

    // Build and run each stage separately
    size_t ctx_size = ggml_tensor_overhead() * 200 + 32 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    fprintf(stderr, "\n=== Testing position embeddings ===\n");
    {
        struct ggml_context * ctx0 = ggml_init(params);
        struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        struct ggml_tensor * output = magpie_build_add_position_embeddings(
            ctx0, input, encoder->pos_emb_w, 0);
        ggml_set_name(output, "output");
        ggml_set_output(output);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 1024, false);
        ggml_build_forward_expand(gf, output);

        ggml_gallocr_reserve(ctx->state.allocr, gf);
        ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

        ggml_backend_tensor_set(input, text_emb.data(), 0, text_emb.size() * sizeof(float));
        ggml_backend_graph_compute(ctx->model.backend, gf);

        std::vector<float> result(d_model * seq_len);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

        compare_tensors("pos_emb", result.data(), enc_with_pos.data(), result.size());
        ggml_free(ctx0);
    }

    fprintf(stderr, "\n=== Testing each encoder layer ===\n");
    std::vector<float> current_input = enc_with_pos;  // Start from input with pos embeddings

    for (int l = 0; l < 6; l++) {
        fprintf(stderr, "\nLayer %d:\n", l);

        struct ggml_context * ctx0 = ggml_init(params);
        struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        // Create causal mask
        struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
        ggml_set_name(mask, "causal_mask");
        ggml_set_input(mask);

        struct ggml_tensor * output = magpie_build_encoder_layer_with_mask(
            ctx0, input, nullptr, &encoder->layers[l], &ctx->model.hparams, mask);
        ggml_set_name(output, "output");
        ggml_set_output(output);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 2048, false);
        ggml_build_forward_expand(gf, output);

        ggml_gallocr_reserve(ctx->state.allocr, gf);
        ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

        // Use reference input (to isolate layer errors)
        ggml_backend_tensor_set(input, current_input.data(), 0, current_input.size() * sizeof(float));

        std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
        fill_causal_mask_f16(mask_data.data(), seq_len);
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        ggml_backend_graph_compute(ctx->model.backend, gf);

        std::vector<float> result(d_model * seq_len);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

        char label[32];
        snprintf(label, sizeof(label), "layer%d (using prev GGML)", l);
        compare_tensors(label, result.data(), layer_expected[l].data(), result.size());

        // Also test using reference input from previous layer
        if (l > 0) {
            ggml_backend_tensor_set(input, layer_expected[l-1].data(), 0, layer_expected[l-1].size() * sizeof(float));
            ggml_backend_graph_compute(ctx->model.backend, gf);
            ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

            snprintf(label, sizeof(label), "layer%d (using ref input)", l);
            compare_tensors(label, result.data(), layer_expected[l].data(), result.size());
        }

        // Update current_input with our output for next layer
        current_input = result;

        ggml_free(ctx0);
    }

    fprintf(stderr, "\n=== Testing final norm ===\n");
    {
        struct ggml_context * ctx0 = ggml_init(params);
        struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        struct ggml_tensor * output = magpie_build_rms_norm(
            ctx0, input, encoder->norm_out_w, ctx->model.hparams.eps);
        ggml_set_name(output, "output");
        ggml_set_output(output);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 1024, false);
        ggml_build_forward_expand(gf, output);

        ggml_gallocr_reserve(ctx->state.allocr, gf);
        ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

        // Use reference layer 5 output
        ggml_backend_tensor_set(input, layer_expected[5].data(), 0, layer_expected[5].size() * sizeof(float));
        ggml_backend_graph_compute(ctx->model.backend, gf);

        std::vector<float> result(d_model * seq_len);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

        compare_tensors("final_norm (ref input)", result.data(), enc_output.data(), result.size());
        ggml_free(ctx0);
    }

    magpie_free(ctx);
    return 0;
}
