#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    if (fread(shape, sizeof(int64_t), 4, f) != 4) { fclose(f); return false; }
    int64_t n = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n);
    if (fread(data.data(), sizeof(float), n, f) != (size_t)n) { fclose(f); return false; }
    fclose(f);
    return true;
}

static void analyze_tensor(const char * name, const float * data, size_t n) {
    float min_val = data[0], max_val = data[0], sum = 0;
    size_t min_idx = 0, max_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] < min_val) { min_val = data[i]; min_idx = i; }
        if (data[i] > max_val) { max_val = data[i]; max_idx = i; }
        sum += data[i];
    }
    fprintf(stderr, "  %s: min=%.4f (at %zu), max=%.4f (at %zu), mean=%.4f\n",
            name, min_val, min_idx, max_val, max_idx, sum / n);
}

static void compare_tensors(const char * name, const float * a, const float * b, size_t n) {
    float max_diff = 0;
    size_t max_idx = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    fprintf(stderr, "  %s: max_diff=%.6f at idx %zu (got=%.6f, exp=%.6f)\n",
            name, max_diff, max_idx, a[max_idx], b[max_idx]);
}

static void fill_causal_mask_f16(ggml_fp16_t * mask, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mask[i * n + j] = (j <= i) ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
}

int main() {
    magpie_context * ctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!ctx) return 1;

    const int d_model = ctx->model.hparams.d_model;
    const int d_ffn = ctx->model.hparams.d_ffn;
    const int enc_heads = ctx->model.hparams.enc_heads;
    const float eps = ctx->model.hparams.eps;

    // Load reference data
    std::vector<float> layer0_out, layer1_out;
    int64_t shape[4];
    read_reference("test_data/reference/manual_enc_layer0_out.bin", layer0_out, shape);
    read_reference("test_data/reference/manual_enc_layer1_out.bin", layer1_out, shape);

    const int seq_len = (int)shape[2];
    fprintf(stderr, "d_model=%d, seq_len=%d, d_ffn=%d\n\n", d_model, seq_len, d_ffn);

    analyze_tensor("layer0_out (ref)", layer0_out.data(), layer0_out.size());
    analyze_tensor("layer1_out (ref)", layer1_out.data(), layer1_out.size());

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    size_t ctx_size = ggml_tensor_overhead() * 200 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Create tensors
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    // Step 1: RMS norm before self-attention
    struct ggml_tensor * norm1 = magpie_build_rms_norm(ctx0, input, layer->norm_self_w, eps);
    ggml_set_name(norm1, "norm1"); ggml_set_output(norm1);

    // Step 2: Self-attention
    struct ggml_tensor * sa = magpie_build_self_attention_with_mask(
        ctx0, norm1, layer->sa_qkv_w, layer->sa_out_w, enc_heads, true, mask);
    ggml_set_name(sa, "self_attn"); ggml_set_output(sa);

    // Step 3: Residual
    struct ggml_tensor * res1 = ggml_add(ctx0, sa, input);
    ggml_set_name(res1, "res1"); ggml_set_output(res1);

    // Step 4: RMS norm before FFN
    struct ggml_tensor * norm2 = magpie_build_rms_norm(ctx0, res1, layer->norm_ff_w, eps);
    ggml_set_name(norm2, "norm2"); ggml_set_output(norm2);

    // Step 5: Conv FFN
    struct ggml_tensor * ffn = magpie_build_conv_ffn(ctx0, norm2, layer->ff_proj_w, layer->ff_out_w, 3);
    ggml_set_name(ffn, "ffn"); ggml_set_output(ffn);

    // Step 6: Residual
    struct ggml_tensor * res2 = ggml_add(ctx0, ffn, res1);
    ggml_set_name(res2, "output"); ggml_set_output(res2);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, res2);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    // Set inputs
    ggml_backend_tensor_set(input, layer0_out.data(), 0, layer0_out.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
    fill_causal_mask_f16(mask_data.data(), seq_len);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    // Compute
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get intermediate results
    fprintf(stderr, "\n=== Intermediate outputs for layer 1 ===\n");
    std::vector<float> buf(d_model * seq_len);

    ggml_backend_tensor_get(norm1, buf.data(), 0, buf.size() * sizeof(float));
    analyze_tensor("norm1", buf.data(), buf.size());

    ggml_backend_tensor_get(sa, buf.data(), 0, buf.size() * sizeof(float));
    analyze_tensor("self_attn", buf.data(), buf.size());

    ggml_backend_tensor_get(res1, buf.data(), 0, buf.size() * sizeof(float));
    analyze_tensor("res1", buf.data(), buf.size());

    ggml_backend_tensor_get(norm2, buf.data(), 0, buf.size() * sizeof(float));
    analyze_tensor("norm2", buf.data(), buf.size());

    ggml_backend_tensor_get(ffn, buf.data(), 0, buf.size() * sizeof(float));
    analyze_tensor("ffn", buf.data(), buf.size());

    ggml_backend_tensor_get(res2, buf.data(), 0, buf.size() * sizeof(float));
    analyze_tensor("output", buf.data(), buf.size());

    fprintf(stderr, "\n=== Comparison to reference ===\n");
    compare_tensors("layer1_out", buf.data(), layer1_out.data(), buf.size());

    // Check index 925 specifically
    fprintf(stderr, "\nIndex 925 (d_model position %d, seq position %d):\n",
            925 % d_model, 925 / d_model);
    fprintf(stderr, "  layer0_out[925] = %.6f\n", layer0_out[925]);
    fprintf(stderr, "  layer1_out[925] expected = %.6f\n", layer1_out[925]);
    fprintf(stderr, "  layer1_out[925] got = %.6f\n", buf[925]);

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
