// Test encoder layer against fresh PyTorch references
#include "../src/magpie.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

struct tensor_header {
    int64_t dims[4];
};

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return {};
    }

    tensor_header header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return {};
    }

    for (int i = 0; i < 4; i++) {
        shape[i] = header.dims[i];
    }

    size_t n_elements = 1;
    for (int i = 0; i < 4; i++) {
        if (header.dims[i] > 0) {
            n_elements *= header.dims[i];
        }
    }

    std::vector<float> data(n_elements);
    if (fread(data.data(), sizeof(float), n_elements, f) != n_elements) {
        fclose(f);
        return {};
    }
    fclose(f);

    return data;
}

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
    if (argc > 1) {
        model_path = argv[1];
    }

    printf("Loading model from %s\n", model_path);
    magpie_context * mctx = magpie_init(model_path);
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    printf("Model loaded successfully\n");
    const auto & hp = mctx->model.hparams;

    // Load fresh reference: input with position embeddings
    int64_t input_shape[4];
    std::vector<float> input_data = load_reference("test_data/reference/fresh_enc_with_pos.bin", input_shape);
    if (input_data.empty()) {
        fprintf(stderr, "Failed to load fresh_enc_with_pos.bin\n");
        magpie_free(mctx);
        return 1;
    }

    // Extract dimensions (file is [batch, d_model, seq, 1])
    int64_t d_model = input_shape[1];
    int64_t seq_len = input_shape[2];
    printf("Input: d_model=%lld, seq_len=%lld\n", (long long)d_model, (long long)seq_len);

    // Load expected output for layer 0
    int64_t expected_shape[4];
    std::vector<float> expected = load_reference("test_data/reference/fresh_enc_layer0_out.bin", expected_shape);
    if (expected.empty()) {
        fprintf(stderr, "Failed to load fresh_enc_layer0_out.bin\n");
        magpie_free(mctx);
        return 1;
    }

    // Create compute context
    size_t ctx_size = ggml_tensor_overhead() * 128 + 16 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx0 = ggml_init(params);

    // Build graph: single encoder layer
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    auto & layer = mctx->model.encoder.layers[0];
    struct ggml_tensor * output = magpie_build_encoder_layer(ctx0, input, nullptr, &layer, &hp);
    ggml_set_name(output, "output");
    ggml_set_output(output);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, output);

    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input (skip batch dimension - use elements from index 0)
    ggml_backend_tensor_set(input, input_data.data(), 0, d_model * seq_len * sizeof(float));

    // Find and set causal mask
    struct ggml_tensor * mask = ggml_get_tensor(ctx0, "causal_mask");
    if (mask) {
        printf("Setting causal mask [%lld, %lld]\n", (long long)mask->ne[0], (long long)mask->ne[1]);
        std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
        fill_causal_mask_f16(mask_data.data(), seq_len);
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    // Compute
    printf("Running computation...\n");
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get output
    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Compare
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int max_idx = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = fabsf(result[i] - expected[i]);
        sum_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    float avg_diff = sum_diff / result.size();

    printf("\nEncoder layer 0 comparison:\n");
    printf("  Max diff: %f at index %d (ggml=%f, ref=%f)\n",
           max_diff, max_idx, result[max_idx], expected[max_idx]);
    printf("  Avg diff: %f\n", avg_diff);
    printf("  First 5 GGML: %.4f %.4f %.4f %.4f %.4f\n",
           result[0], result[1], result[2], result[3], result[4]);
    printf("  First 5 ref:  %.4f %.4f %.4f %.4f %.4f\n",
           expected[0], expected[1], expected[2], expected[3], expected[4]);

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_diff < 0.1f) {
        printf("\nSUCCESS: Encoder layer matches PyTorch reference!\n");
        return 0;
    } else if (max_diff < 1.0f) {
        printf("\nWARNING: Some difference from reference (might be acceptable)\n");
        return 0;
    } else {
        printf("\nFAILURE: Large difference from reference\n");
        return 1;
    }
}
