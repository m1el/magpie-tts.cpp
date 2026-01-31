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
    ggml_set_name(attn_out, "attn_raw"); ggml_set_output(attn_out);

    struct ggml_tensor * attn_reshaped = ggml_cont(ctx0, attn_out);
    attn_reshaped = ggml_reshape_2d(ctx0, attn_reshaped, d_model, seq_len);
    ggml_set_name(attn_reshaped, "attn_reshaped"); ggml_set_output(attn_reshaped);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, attn_reshaped);
    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);
    ggml_backend_tensor_set(input, layer0_out.data(), 0, layer0_out.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
    fill_mask(mask_data.data(), seq_len);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get raw flash attention output [d_head, n_heads, seq]
    std::vector<float> raw_attn(d_head * n_heads * seq_len);
    ggml_backend_tensor_get(ggml_get_tensor(ctx0, "attn_raw"), raw_attn.data(), 0, raw_attn.size() * sizeof(float));

    std::vector<float> got_attn(d_model * seq_len);
    ggml_backend_tensor_get(attn_reshaped, got_attn.data(), 0, got_attn.size() * sizeof(float));

    // Focus on token 9, heads 10 and 11
    int t = 9;
    fprintf(stderr, "Flash attention raw output is [%lld, %lld, %lld]\n",
            (long long)attn_out->ne[0], (long long)attn_out->ne[1], (long long)attn_out->ne[2]);

    for (int h = 10; h <= 11; h++) {
        fprintf(stderr, "\n=== Head %d, token %d ===\n", h, t);

        // In raw [d_head, n_heads, seq] = [64, 12, 14]:
        // Element [i, h, t] at offset i + h * 64 + t * 768
        fprintf(stderr, "Raw flash output (first 5 dims of head %d, token %d):\n", h, t);
        for (int i = 0; i < 5; i++) {
            int raw_offset = i + h * d_head + t * d_model;
            fprintf(stderr, "  i=%d: raw[%d]=%+.6f\n", i, raw_offset, raw_attn[raw_offset]);
        }

        // In reshaped [d_model, seq]:
        // Element [d, t] at offset d + t * d_model where d = h * 64 + i
        fprintf(stderr, "\nReshaped output (dims %d-%d, token %d):\n", h*64, h*64+4, t);
        for (int i = 0; i < 5; i++) {
            int d = h * d_head + i;
            int ggml_offset = d + t * d_model;
            int ref_offset = t * d_model + d;  // Same because t*768+d = d+t*768
            fprintf(stderr, "  d=%d: ggml[%d]=%+.6f, ref[%d]=%+.6f\n",
                    d, ggml_offset, got_attn[ggml_offset], ref_offset, ref_attn_out[ref_offset]);
        }

        // Check if maybe head 11 values appear at a different position
        fprintf(stderr, "\nSearching for ref value ref_attn_out[t=9, d=%d]=%.6f in GGML:\n",
                h*64, ref_attn_out[t * d_model + h * d_head]);
        float target = ref_attn_out[t * d_model + h * d_head];
        for (int d = 0; d < d_model; d++) {
            if (std::fabs(got_attn[d + t * d_model] - target) < 0.001) {
                fprintf(stderr, "  Found at d=%d (head %d, i=%d)\n", d, d / 64, d % 64);
            }
        }
    }

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
