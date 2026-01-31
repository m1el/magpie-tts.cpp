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

int main() {
    magpie_context * ctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!ctx) return 1;

    const int d_model = ctx->model.hparams.d_model;
    const float eps = ctx->model.hparams.eps;
    const int n_heads = ctx->model.hparams.enc_heads;
    const int d_head = d_model / n_heads;

    std::vector<float> layer0_out;
    int64_t shape[4];
    read_ref("test_data/reference/manual_enc_layer0_out.bin", layer0_out, shape);
    const int seq_len = (int)shape[2];

    // Load PyTorch attention output reference
    std::vector<float> ref_attn_out;
    int64_t attn_shape[4];
    if (!read_ref("test_data/reference/debug_l1_attn_out.bin", ref_attn_out, attn_shape)) {
        fprintf(stderr, "Failed to load ref attn out\n");
        return 1;
    }
    fprintf(stderr, "ref_attn_out shape: [%lld, %lld, %lld, %lld]\n",
            (long long)attn_shape[0], (long long)attn_shape[1],
            (long long)attn_shape[2], (long long)attn_shape[3]);

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    size_t ctx_size = ggml_tensor_overhead() * 200 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    // Build same graph as debug_attention
    struct ggml_tensor * norm1 = magpie_build_rms_norm(ctx0, input, layer->norm_self_w, eps);
    struct ggml_tensor * qkv = ggml_mul_mat(ctx0, layer->sa_qkv_w, norm1);
    struct ggml_tensor * q = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], 0);
    struct ggml_tensor * k = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], d_model * sizeof(float));
    struct ggml_tensor * v = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], 2 * d_model * sizeof(float));

    q = ggml_cont(ctx0, q);
    k = ggml_cont(ctx0, k);
    v = ggml_cont(ctx0, v);

    q = ggml_reshape_3d(ctx0, q, d_head, n_heads, seq_len);
    k = ggml_reshape_3d(ctx0, k, d_head, n_heads, seq_len);
    v = ggml_reshape_3d(ctx0, v, d_head, n_heads, seq_len);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    q = ggml_cont(ctx0, q);
    k = ggml_cont(ctx0, k);
    v = ggml_cont(ctx0, v);

    float scale = 1.0f / sqrtf((float)d_head);
    struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx0, q, k, v, mask, scale, 0.0f, 0.0f);

    attn_out = ggml_cont(ctx0, attn_out);
    attn_out = ggml_reshape_2d(ctx0, attn_out, d_model, seq_len);
    ggml_set_name(attn_out, "attn_reshaped"); ggml_set_output(attn_out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, attn_out);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    ggml_backend_tensor_set(input, layer0_out.data(), 0, layer0_out.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
    fill_mask(mask_data.data(), seq_len);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(ctx->model.backend, gf);

    std::vector<float> got_attn(d_model * seq_len);
    ggml_backend_tensor_get(attn_out, got_attn.data(), 0, got_attn.size() * sizeof(float));

    // Compare memory layouts
    fprintf(stderr, "\n=== Memory layout comparison ===\n");
    fprintf(stderr, "GGML attn_reshaped is [%lld, %lld] (col-major)\n",
            (long long)attn_out->ne[0], (long long)attn_out->ne[1]);
    fprintf(stderr, "PyTorch ref_attn_out is [seq=%d, d_model=%d] (row-major)\n", seq_len, d_model);

    // In GGML col-major [768, 14]: element [d, t] at offset d + t*768
    // In PyTorch row-major [14, 768]: element [t, d] at offset t*768 + d
    // These are the SAME memory offsets!

    fprintf(stderr, "\nFirst 10 memory offsets:\n");
    for (int i = 0; i < 10; i++) {
        fprintf(stderr, "offset %d: GGML=%.6f, PyTorch=%.6f, diff=%.6f\n",
                i, got_attn[i], ref_attn_out[i], std::fabs(got_attn[i] - ref_attn_out[i]));
    }

    fprintf(stderr, "\nToken 0, first 5 dims (should match):\n");
    for (int d = 0; d < 5; d++) {
        int offset = d;  // GGML [d, 0] = PyTorch [0, d] in memory
        fprintf(stderr, "  d=%d: GGML=%.6f, PyTorch=%.6f\n", d, got_attn[offset], ref_attn_out[offset]);
    }

    fprintf(stderr, "\nToken 9, first 5 dims:\n");
    for (int d = 0; d < 5; d++) {
        int offset = d + 9 * d_model;
        fprintf(stderr, "  d=%d: GGML=%.6f, PyTorch=%.6f\n", d, got_attn[offset], ref_attn_out[offset]);
    }

    fprintf(stderr, "\nPosition 7322 (token 9, dim 410):\n");
    fprintf(stderr, "  GGML=%.6f, PyTorch=%.6f\n", got_attn[7322], ref_attn_out[7322]);

    // Now compute the row sums (sum over d for each t) - should be similar
    fprintf(stderr, "\nRow sums (sum over d for each t):\n");
    for (int t = 0; t < 3; t++) {
        float sum_ggml = 0, sum_pytorch = 0;
        for (int d = 0; d < d_model; d++) {
            int offset = d + t * d_model;
            sum_ggml += got_attn[offset];
            sum_pytorch += ref_attn_out[offset];
        }
        fprintf(stderr, "  t=%d: GGML=%.6f, PyTorch=%.6f, diff=%.6f\n",
                t, sum_ggml, sum_pytorch, std::fabs(sum_ggml - sum_pytorch));
    }

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
