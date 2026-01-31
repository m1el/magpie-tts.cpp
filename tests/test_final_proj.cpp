// Test final projection layer against PyTorch reference
#include "../src/magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return {};
    }
    int64_t dims[4];
    if (fread(dims, sizeof(int64_t), 4, f) != 4) { fclose(f); return {}; }
    for (int i = 0; i < 4; i++) shape[i] = dims[i];
    size_t n = 1;
    for (int i = 0; i < 4; i++) if (dims[i] > 0) n *= dims[i];
    std::vector<float> data(n);
    if (fread(data.data(), sizeof(float), n, f) != n) { fclose(f); return {}; }
    fclose(f);
    return data;
}

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = mctx->model.hparams;

    // Load reference data
    int64_t dec_out_shape[4], logits_shape[4];
    auto dec_out_data = load_reference("test_data/reference/dec_output.bin", dec_out_shape);
    auto logits_expected = load_reference("test_data/reference/dec_logits.bin", logits_shape);

    if (dec_out_data.empty() || logits_expected.empty()) {
        fprintf(stderr, "Failed to load reference data. Run: uv run scripts/dump_decoder_reference.py\n");
        magpie_free(mctx);
        return 1;
    }

    int64_t d_model = dec_out_shape[0];
    int64_t dec_seq = dec_out_shape[1];
    int64_t logits_dim = logits_shape[0];  // 16192 = 8 * 2024

    printf("d_model=%lld, dec_seq=%lld, logits_dim=%lld\n",
           (long long)d_model, (long long)dec_seq, (long long)logits_dim);
    printf("Expected logits: 8 codebooks * %d vocab = %d\n",
           hp.vocab_per_cb, hp.num_codebooks * hp.vocab_per_cb);

    // Extract last frame of decoder output
    // dec_output is [d_model, dec_seq] in column-major order
    // Last frame is at offset (dec_seq-1) * d_model
    std::vector<float> last_frame(d_model);
    for (int i = 0; i < d_model; i++) {
        last_frame[i] = dec_out_data[(dec_seq - 1) * d_model + i];
    }

    printf("Last frame first5: %.6f %.6f %.6f %.6f %.6f\n",
           last_frame[0], last_frame[1], last_frame[2], last_frame[3], last_frame[4]);

    // Build graph
    size_t ctx_size = ggml_tensor_overhead() * 64 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensor (single frame)
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_model);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    // Build final projection
    struct ggml_tensor * output = magpie_build_final_proj(ctx0, input, &mctx->model.final_proj);

    if (!output) {
        fprintf(stderr, "Failed to build final projection\n");
        ggml_free(ctx0);
        magpie_free(mctx);
        return 1;
    }

    ggml_set_name(output, "output");
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, output);
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate and run
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data
    ggml_backend_tensor_set(input, last_frame.data(), 0, d_model * sizeof(float));

    printf("Computing final projection...\n");
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get result
    std::vector<float> result(logits_dim);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Compare
    float max_diff = 0;
    int max_idx = 0;
    float sum_diff = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = fabsf(result[i] - logits_expected[i]);
        sum_diff += diff;
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }

    printf("\n=== Final Projection Comparison ===\n");
    printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
           result[0], result[1], result[2], result[3], result[4]);
    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           logits_expected[0], logits_expected[1], logits_expected[2],
           logits_expected[3], logits_expected[4]);
    printf("  GGML last5:     %.6f %.6f %.6f %.6f %.6f\n",
           result[logits_dim-5], result[logits_dim-4], result[logits_dim-3],
           result[logits_dim-2], result[logits_dim-1]);
    printf("  PyTorch last5:  %.6f %.6f %.6f %.6f %.6f\n",
           logits_expected[logits_dim-5], logits_expected[logits_dim-4],
           logits_expected[logits_dim-3], logits_expected[logits_dim-2],
           logits_expected[logits_dim-1]);
    printf("  Max diff: %.6f at idx %d\n", max_diff, max_idx);
    printf("  Avg diff: %.6f\n", sum_diff / result.size());

    // Show which codebook the max diff is in
    int cb_idx = max_idx / hp.vocab_per_cb;
    int token_idx = max_idx % hp.vocab_per_cb;
    printf("  Max diff at codebook %d, token %d\n", cb_idx, token_idx);

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_diff < 0.01f) {
        printf("\nSUCCESS: Final projection matches within tolerance!\n");
        return 0;
    } else {
        printf("\nFAILURE: Final projection has large errors\n");
        return 1;
    }
}
