// Test local transformer step 0 against PyTorch reference
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

static void fill_causal_mask_f32(float * mask, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            mask[i * seq_len + j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }
}

// Declare functions (implemented in magpie.cpp but not in header)
extern ggml_tensor * magpie_build_local_transformer_step0(
    ggml_context * ctx,
    ggml_tensor * decoder_hidden,
    magpie_local_transformer * lt,
    const magpie_hparams * hparams);

extern std::vector<int32_t> magpie_local_transformer_sample_all(
    magpie_context * mctx,
    const float * decoder_hidden_data,
    float temperature,
    int top_k);

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = mctx->model.hparams;

    // Load reference data
    int64_t hidden_shape[4], projected_shape[4], logits_shape[4];
    auto hidden_data = load_reference("test_data/reference/lt_dec_hidden.bin", hidden_shape);
    auto projected_expected = load_reference("test_data/reference/lt_input_projected.bin", projected_shape);
    auto logits_expected = load_reference("test_data/reference/lt_logits_cb0.bin", logits_shape);

    if (hidden_data.empty() || projected_expected.empty() || logits_expected.empty()) {
        fprintf(stderr, "Failed to load reference data. Run: uv run scripts/dump_local_transformer_reference.py\n");
        magpie_free(mctx);
        return 1;
    }

    int64_t d_model = hidden_shape[0];
    int64_t lt_dim = projected_shape[0];
    int64_t vocab_per_cb = logits_shape[0];

    printf("d_model=%lld, lt_dim=%lld, vocab_per_cb=%lld\n",
           (long long)d_model, (long long)lt_dim, (long long)vocab_per_cb);

    // ================================================================
    // Test 1: Input projection only
    // ================================================================
    printf("\n=== Test 1: Input Projection ===\n");
    {
        size_t ctx_size = ggml_tensor_overhead() * 32 + 8 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_model);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        // Just input projection
        struct ggml_tensor * output = ggml_mul_mat(ctx0, mctx->model.local_transformer.in_proj_w, input);
        output = ggml_add(ctx0, output, mctx->model.local_transformer.in_proj_b);
        ggml_set_name(output, "output");
        ggml_set_output(output);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, output);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(input, hidden_data.data(), 0, d_model * sizeof(float));
        ggml_backend_graph_compute(mctx->model.backend, gf);

        std::vector<float> result(lt_dim);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

        float max_diff = 0;
        for (size_t i = 0; i < result.size(); i++) {
            float diff = fabsf(result[i] - projected_expected[i]);
            if (diff > max_diff) max_diff = diff;
        }

        printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
               result[0], result[1], result[2], result[3], result[4]);
        printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
               projected_expected[0], projected_expected[1], projected_expected[2],
               projected_expected[3], projected_expected[4]);
        printf("  Max diff: %.6f\n", max_diff);

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);

        if (max_diff > 0.001f) {
            printf("  FAILED: Input projection has large errors\n");
            magpie_free(mctx);
            return 1;
        }
        printf("  PASSED\n");
    }

    // ================================================================
    // Test 2: Full local transformer step 0 (codebook 0 logits)
    // ================================================================
    printf("\n=== Test 2: Full Local Transformer Step 0 ===\n");
    {
        size_t ctx_size = ggml_tensor_overhead() * 256 + 64 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_model);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        struct ggml_tensor * output = magpie_build_local_transformer_step0(
            ctx0, input, &mctx->model.local_transformer, &hp);

        if (!output) {
            fprintf(stderr, "Failed to build local transformer\n");
            ggml_free(ctx0);
            magpie_free(mctx);
            return 1;
        }

        ggml_set_name(output, "output");
        ggml_set_output(output);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
        ggml_build_forward_expand(gf, output);
        printf("  Graph nodes: %d\n", ggml_graph_n_nodes(gf));

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(input, hidden_data.data(), 0, d_model * sizeof(float));

        // Set causal mask (1x1 for single position)
        struct ggml_tensor * mask = ggml_get_tensor(ctx0, "lt_causal_mask");
        if (mask) {
            printf("  Setting causal mask [%lld, %lld]\n", (long long)mask->ne[0], (long long)mask->ne[1]);
            std::vector<float> mask_data(mask->ne[0] * mask->ne[1]);
            fill_causal_mask_f32(mask_data.data(), mask->ne[0]);
            ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
        }

        printf("  Computing local transformer step 0...\n");
        ggml_backend_graph_compute(mctx->model.backend, gf);

        std::vector<float> result(vocab_per_cb);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

        float max_diff = 0;
        int max_idx = 0;
        float sum_diff = 0;
        for (size_t i = 0; i < result.size(); i++) {
            float diff = fabsf(result[i] - logits_expected[i]);
            sum_diff += diff;
            if (diff > max_diff) { max_diff = diff; max_idx = i; }
        }

        printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
               result[0], result[1], result[2], result[3], result[4]);
        printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
               logits_expected[0], logits_expected[1], logits_expected[2],
               logits_expected[3], logits_expected[4]);

        // Find argmax
        int ggml_argmax = 0, pytorch_argmax = 0;
        float ggml_max = result[0], pytorch_max = logits_expected[0];
        for (size_t i = 1; i < result.size(); i++) {
            if (result[i] > ggml_max) { ggml_max = result[i]; ggml_argmax = i; }
            if (logits_expected[i] > pytorch_max) { pytorch_max = logits_expected[i]; pytorch_argmax = i; }
        }
        printf("  GGML argmax:    %d (value: %.6f)\n", ggml_argmax, ggml_max);
        printf("  PyTorch argmax: %d (value: %.6f)\n", pytorch_argmax, pytorch_max);
        printf("  Max diff: %.6f at idx %d\n", max_diff, max_idx);
        printf("  Avg diff: %.6f\n", sum_diff / result.size());

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);

        if (max_diff > 0.01f) {
            printf("  FAILED: Local transformer has large errors\n");
            magpie_free(mctx);
            return 1;
        }
        if (ggml_argmax != pytorch_argmax) {
            printf("  WARNING: Argmax mismatch (may affect sampling)\n");
        }
        printf("  PASSED\n");
    }

    // ================================================================
    // Test 3: Full autoregressive sampling (all 8 codebooks)
    // ================================================================
    printf("\n=== Test 3: Full Autoregressive Sampling ===\n");
    {
        // Expected codes from PyTorch reference (argmax sampling)
        int expected_codes[8] = {293, 1454, 512, 1455, 476, 40, 1817, 1014};

        printf("  Running full local transformer...\n");
        std::vector<int32_t> sampled = magpie_local_transformer_sample_all(
            mctx, hidden_data.data(), 0.0f, 0);

        printf("  GGML sampled:    ");
        for (int i = 0; i < 8; i++) printf("%d ", sampled[i]);
        printf("\n");

        printf("  PyTorch sampled: ");
        for (int i = 0; i < 8; i++) printf("%d ", expected_codes[i]);
        printf("\n");

        int mismatches = 0;
        for (int i = 0; i < 8; i++) {
            if (sampled[i] != expected_codes[i]) {
                printf("  MISMATCH at codebook %d: GGML=%d, PyTorch=%d\n",
                       i, sampled[i], expected_codes[i]);
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("  PASSED - All 8 codebooks match!\n");
        } else {
            printf("  FAILED - %d mismatches\n", mismatches);
            magpie_free(mctx);
            return 1;
        }
    }

    magpie_free(mctx);
    printf("\n=== All Local Transformer Tests Passed ===\n");
    return 0;
}
