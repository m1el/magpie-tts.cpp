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

    // Load reference data
    std::vector<float> layer0_out;
    int64_t shape[4];
    read_ref("test_data/reference/manual_enc_layer0_out.bin", layer0_out, shape);
    const int seq_len = (int)shape[2];

    // Load PyTorch QKV reference if available
    std::vector<float> ref_qkv;
    int64_t qkv_shape[4];
    bool has_qkv = read_ref("test_data/reference/debug_l1_qkv.bin", ref_qkv, qkv_shape);

    fprintf(stderr, "d_model=%d, seq_len=%d, n_heads=%d, d_head=%d\n", d_model, seq_len, n_heads, d_head);

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    size_t ctx_size = ggml_tensor_overhead() * 200 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Build graph with intermediate attention outputs
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    // Norm
    struct ggml_tensor * norm1 = magpie_build_rms_norm(ctx0, input, layer->norm_self_w, eps);
    ggml_set_name(norm1, "norm1"); ggml_set_output(norm1);

    // QKV projection
    struct ggml_tensor * qkv = ggml_mul_mat(ctx0, layer->sa_qkv_w, norm1);
    ggml_set_name(qkv, "qkv"); ggml_set_output(qkv);

    // Split Q, K, V
    struct ggml_tensor * q = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], 0);
    struct ggml_tensor * k = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], d_model * sizeof(float));
    struct ggml_tensor * v = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], 2 * d_model * sizeof(float));

    q = ggml_cont(ctx0, q); ggml_set_name(q, "q"); ggml_set_output(q);
    k = ggml_cont(ctx0, k);
    v = ggml_cont(ctx0, v);

    // Reshape for multi-head: [d_head, n_heads, seq]
    q = ggml_reshape_3d(ctx0, q, d_head, n_heads, seq_len);
    k = ggml_reshape_3d(ctx0, k, d_head, n_heads, seq_len);
    v = ggml_reshape_3d(ctx0, v, d_head, n_heads, seq_len);

    // Permute for flash attention: [d_head, seq, n_heads]
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    q = ggml_cont(ctx0, q);
    k = ggml_cont(ctx0, k);
    v = ggml_cont(ctx0, v);

    fprintf(stderr, "Q shape before flash attn: [%lld, %lld, %lld]\\n",
            (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2]);

    // Flash attention
    float scale = 1.0f / sqrtf((float)d_head);
    struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx0, q, k, v, mask, scale, 0.0f, 0.0f);
    ggml_set_name(attn_out, "attn_out"); ggml_set_output(attn_out);

    fprintf(stderr, "Flash attention output shape: [%lld, %lld, %lld, %lld]\\n",
            (long long)attn_out->ne[0], (long long)attn_out->ne[1],
            (long long)attn_out->ne[2], (long long)attn_out->ne[3]);

    // Reshape back to [d_model, seq]
    attn_out = ggml_cont(ctx0, attn_out);
    attn_out = ggml_reshape_2d(ctx0, attn_out, d_model, seq_len);
    ggml_set_name(attn_out, "attn_reshaped"); ggml_set_output(attn_out);

    // Output projection
    struct ggml_tensor * sa_out = ggml_mul_mat(ctx0, layer->sa_out_w, attn_out);
    ggml_set_name(sa_out, "sa_out"); ggml_set_output(sa_out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, sa_out);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    ggml_backend_tensor_set(input, layer0_out.data(), 0, layer0_out.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
    fill_mask(mask_data.data(), seq_len);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get results
    std::vector<float> got_qkv(3 * d_model * seq_len);
    std::vector<float> got_q(d_model * seq_len);
    std::vector<float> got_attn_reshaped(d_model * seq_len);
    std::vector<float> got_sa(d_model * seq_len);

    ggml_backend_tensor_get(qkv, got_qkv.data(), 0, got_qkv.size() * sizeof(float));
    ggml_backend_tensor_get(ggml_get_tensor(ctx0, "q"), got_q.data(), 0, got_q.size() * sizeof(float));
    ggml_backend_tensor_get(ggml_get_tensor(ctx0, "attn_reshaped"), got_attn_reshaped.data(), 0, got_attn_reshaped.size() * sizeof(float));
    ggml_backend_tensor_get(sa_out, got_sa.data(), 0, got_sa.size() * sizeof(float));

    // Compare QKV
    if (has_qkv) {
        fprintf(stderr, "\n=== QKV comparison ===\n");
        float max_diff = 0;
        size_t max_idx = 0;
        for (size_t i = 0; i < got_qkv.size(); i++) {
            float diff = std::fabs(got_qkv[i] - ref_qkv[i]);
            if (diff > max_diff) { max_diff = diff; max_idx = i; }
        }
        fprintf(stderr, "QKV max_diff=%.6f at %zu (got=%.4f, exp=%.4f)\n",
                max_diff, max_idx, got_qkv[max_idx], ref_qkv[max_idx]);
    }

    // Position 7322 = token 9, dim 410
    int pos = 7322;
    int t = pos / 768;  // token 9
    int d = pos % 768;  // dim 410
    fprintf(stderr, "\n=== Position %d (token %d, dim %d) ===\n", pos, t, d);
    fprintf(stderr, "input[%d] = %.6f\n", pos, layer0_out[pos]);

    // In GGML [d_model, seq] col-major: element [d, t] = data[d + t*d_model]
    // So element (d=410, t=9) is at offset 410 + 9*768 = 7322
    fprintf(stderr, "q[%d] = %.6f\n", pos, got_q[pos]);
    fprintf(stderr, "attn_reshaped[%d] = %.6f (PyTorch: 0.045048)\n", pos, got_attn_reshaped[pos]);
    fprintf(stderr, "sa_out[%d] = %.6f (PyTorch: 0.246909)\n", pos, got_sa[pos]);

    // Print Q values for first head at token 9
    fprintf(stderr, "\nQ values for token 9, head 0 (first 5 dims):\n");
    for (int i = 0; i < 5; i++) {
        // In GGML q after reshape is [d_head, n_heads, seq] col-major
        // Element [i, 0, 9] = data[i + 0*d_head + 9*d_head*n_heads]
        // But actually after cont, the layout might be different...
        fprintf(stderr, "  q[%d, 0, 9] = %.6f\n", i, got_q[i + 9*768]);
    }

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
