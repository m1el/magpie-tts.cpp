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

    std::vector<float> ref_attn_out;
    int64_t attn_shape[4];
    read_ref("test_data/reference/debug_l1_attn_out.bin", ref_attn_out, attn_shape);

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    std::vector<float> weight(d_model * d_model);
    ggml_backend_tensor_get(layer->sa_out_w, weight.data(), 0, weight.size() * sizeof(float));

    // Compute GGML attention
    size_t ctx_size = ggml_tensor_overhead() * 200 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    struct ggml_tensor * norm1 = magpie_build_rms_norm(ctx0, input, layer->norm_self_w, eps);
    struct ggml_tensor * qkv = ggml_mul_mat(ctx0, layer->sa_qkv_w, norm1);
    struct ggml_tensor * q = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], 0);
    struct ggml_tensor * k = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], d_model * sizeof(float));
    struct ggml_tensor * v = ggml_view_2d(ctx0, qkv, d_model, seq_len, qkv->nb[1], 2 * d_model * sizeof(float));

    q = ggml_cont(ctx0, q);  k = ggml_cont(ctx0, k);  v = ggml_cont(ctx0, v);
    q = ggml_reshape_3d(ctx0, q, d_head, n_heads, seq_len);
    k = ggml_reshape_3d(ctx0, k, d_head, n_heads, seq_len);
    v = ggml_reshape_3d(ctx0, v, d_head, n_heads, seq_len);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    q = ggml_cont(ctx0, q);  k = ggml_cont(ctx0, k);  v = ggml_cont(ctx0, v);

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

    // Compare per-head sums for token 9
    int t = 9, o = 410;
    fprintf(stderr, "Per-head breakdown for matmul at (t=%d, o=%d):\n\n", t, o);

    float total_ggml = 0, total_ref = 0;
    for (int h = 0; h < n_heads; h++) {
        float head_sum_ggml = 0, head_sum_ref = 0;
        for (int i = 0; i < d_head; i++) {
            int k = h * d_head + i;  // PyTorch-style: head h, dim i gives k = h * 64 + i
            float w = weight[k + o * d_model];
            float a_ggml = got_attn[k + t * d_model];
            float a_ref = ref_attn_out[t * d_model + k];
            head_sum_ggml += w * a_ggml;
            head_sum_ref += w * a_ref;
        }
        total_ggml += head_sum_ggml;
        total_ref += head_sum_ref;
        fprintf(stderr, "Head %2d: sum_ggml=%.6f, sum_ref=%.6f, diff=%.6f, ratio=%.3f\n",
                h, head_sum_ggml, head_sum_ref, head_sum_ggml - head_sum_ref,
                head_sum_ref != 0 ? head_sum_ggml / head_sum_ref : 0);
    }
    fprintf(stderr, "\nTotal: GGML=%.6f, ref=%.6f, ratio=%.3f\n",
            total_ggml, total_ref, total_ggml / total_ref);

    // Also check the raw attention values per head
    fprintf(stderr, "\n=== Attention output per head for token 9 ===\n");
    for (int h = 0; h < n_heads; h++) {
        float sum_ggml = 0, sum_ref = 0;
        for (int i = 0; i < d_head; i++) {
            int k = h * d_head + i;
            sum_ggml += got_attn[k + t * d_model];
            sum_ref += ref_attn_out[t * d_model + k];
        }
        fprintf(stderr, "Head %2d: attn_sum_ggml=%.4f, attn_sum_ref=%.4f, diff=%.4f\n",
                h, sum_ggml, sum_ref, sum_ggml - sum_ref);
    }

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
