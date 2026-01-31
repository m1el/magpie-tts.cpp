#include "magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

static bool read_ref(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    if (fread(shape, sizeof(int64_t), 4, f) != 4) { fclose(f); return false; }
    int64_t n = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n);
    if (fread(data.data(), sizeof(float), n, f) != (size_t)n) { fclose(f); return false; }
    fclose(f);
    return true;
}

static void fill_mask(ggml_fp16_t * mask, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mask[i*n + j] = (j <= i) ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
}

static void compare(const char * name, const float * got, const float * exp, size_t n) {
    float max_diff = 0;
    size_t max_idx = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(got[i] - exp[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    fprintf(stderr, "%s: max_diff=%.6f at %zu (got=%.4f, exp=%.4f)\n",
            name, max_diff, max_idx, got[max_idx], exp[max_idx]);
}

int main() {
    magpie_context * ctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!ctx) return 1;

    const int d_model = ctx->model.hparams.d_model;
    const float eps = ctx->model.hparams.eps;
    const int enc_heads = ctx->model.hparams.enc_heads;
    const int enc_kernel = ctx->model.hparams.enc_kernel;

    // Load reference data
    std::vector<float> layer0_out, layer1_out;
    std::vector<float> ref_norm1, ref_sa, ref_res1, ref_norm2, ref_ffn;
    int64_t shape[4];

    read_ref("test_data/reference/manual_enc_layer0_out.bin", layer0_out, shape);
    read_ref("test_data/reference/manual_enc_layer1_out.bin", layer1_out, shape);

    // Try to load detailed PyTorch debug data if available
    bool has_debug = read_ref("test_data/reference/debug_l1_norm1.bin", ref_norm1, shape);
    if (has_debug) {
        read_ref("test_data/reference/debug_l1_sa.bin", ref_sa, shape);
        read_ref("test_data/reference/debug_l1_res1.bin", ref_res1, shape);
        read_ref("test_data/reference/debug_l1_norm2.bin", ref_norm2, shape);
        read_ref("test_data/reference/debug_l1_ffn_correct.bin", ref_ffn, shape);
        fprintf(stderr, "Loaded PyTorch debug data\n");
    }

    const int seq_len = (int)shape[2];
    fprintf(stderr, "d_model=%d, seq_len=%d\n\n", d_model, seq_len);

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    size_t ctx_size = ggml_tensor_overhead() * 200 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Build compute graph with intermediate outputs
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    // Step 1: norm1
    struct ggml_tensor * norm1 = magpie_build_rms_norm(ctx0, input, layer->norm_self_w, eps);
    ggml_set_name(norm1, "norm1"); ggml_set_output(norm1);

    // Step 2: self-attention
    struct ggml_tensor * sa = magpie_build_self_attention_with_mask(
        ctx0, norm1, layer->sa_qkv_w, layer->sa_out_w, enc_heads, true, mask);
    ggml_set_name(sa, "sa"); ggml_set_output(sa);

    // Step 3: residual 1
    struct ggml_tensor * res1 = ggml_add(ctx0, sa, input);
    ggml_set_name(res1, "res1"); ggml_set_output(res1);

    // Step 4: norm2
    struct ggml_tensor * norm2 = magpie_build_rms_norm(ctx0, res1, layer->norm_ff_w, eps);
    ggml_set_name(norm2, "norm2"); ggml_set_output(norm2);

    // Step 5: FFN
    struct ggml_tensor * ffn = magpie_build_conv_ffn(ctx0, norm2, layer->ff_proj_w, layer->ff_out_w, enc_kernel);
    ggml_set_name(ffn, "ffn"); ggml_set_output(ffn);

    // Step 6: residual 2 (final output)
    struct ggml_tensor * output = ggml_add(ctx0, ffn, res1);
    ggml_set_name(output, "output"); ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    ggml_backend_tensor_set(input, layer0_out.data(), 0, layer0_out.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
    fill_mask(mask_data.data(), seq_len);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get results
    std::vector<float> got_norm1(d_model * seq_len);
    std::vector<float> got_sa(d_model * seq_len);
    std::vector<float> got_res1(d_model * seq_len);
    std::vector<float> got_norm2(d_model * seq_len);
    std::vector<float> got_ffn(d_model * seq_len);
    std::vector<float> got_output(d_model * seq_len);

    ggml_backend_tensor_get(norm1, got_norm1.data(), 0, got_norm1.size() * sizeof(float));
    ggml_backend_tensor_get(sa, got_sa.data(), 0, got_sa.size() * sizeof(float));
    ggml_backend_tensor_get(res1, got_res1.data(), 0, got_res1.size() * sizeof(float));
    ggml_backend_tensor_get(norm2, got_norm2.data(), 0, got_norm2.size() * sizeof(float));
    ggml_backend_tensor_get(ffn, got_ffn.data(), 0, got_ffn.size() * sizeof(float));
    ggml_backend_tensor_get(output, got_output.data(), 0, got_output.size() * sizeof(float));

    fprintf(stderr, "=== Comparison with PyTorch debug data ===\n");
    if (has_debug) {
        compare("norm1", got_norm1.data(), ref_norm1.data(), got_norm1.size());
        compare("sa   ", got_sa.data(), ref_sa.data(), got_sa.size());
        compare("res1 ", got_res1.data(), ref_res1.data(), got_res1.size());
        compare("norm2", got_norm2.data(), ref_norm2.data(), got_norm2.size());
        compare("ffn  ", got_ffn.data(), ref_ffn.data(), got_ffn.size());
    }
    compare("output", got_output.data(), layer1_out.data(), got_output.size());

    // Check position 925 specifically
    fprintf(stderr, "\n=== Position 925 (token 1, dim 157) ===\n");
    fprintf(stderr, "input[925]  = %.6f\n", layer0_out[925]);
    fprintf(stderr, "norm1[925]  = %.6f\n", got_norm1[925]);
    fprintf(stderr, "sa[925]     = %.6f\n", got_sa[925]);
    fprintf(stderr, "res1[925]   = %.6f\n", got_res1[925]);
    fprintf(stderr, "norm2[925]  = %.6f\n", got_norm2[925]);
    fprintf(stderr, "ffn[925]    = %.6f\n", got_ffn[925]);
    fprintf(stderr, "output[925] = %.6f (expected: %.6f)\n", got_output[925], layer1_out[925]);

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
