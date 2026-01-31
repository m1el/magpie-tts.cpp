// Debug decoder layer components step by step
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

static void compare_tensors(const char* name, const float* ggml_data, const float* ref_data, size_t n) {
    float max_diff = 0;
    int max_idx = 0;
    float sum_diff = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ggml_data[i] - ref_data[i]);
        sum_diff += diff;
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    printf("%s: max_diff=%.6f at idx %d, avg_diff=%.6f\n",
           name, max_diff, max_idx, sum_diff / n);
    printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
           ggml_data[0], ggml_data[1], ggml_data[2], ggml_data[3], ggml_data[4]);
    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           ref_data[0], ref_data[1], ref_data[2], ref_data[3], ref_data[4]);
}

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = mctx->model.hparams;
    auto & layer = mctx->model.decoder.layers[0];

    // Load reference data
    int64_t dec_input_shape[4], enc_out_shape[4];
    auto dec_input_data = load_reference("test_data/reference/dec_input.bin", dec_input_shape);
    auto enc_out_data = load_reference("test_data/reference/dec_encoder_output.bin", enc_out_shape);

    // Load intermediate reference data
    int64_t shape[4];
    auto norm_self_ref = load_reference("test_data/reference/dec_l0_norm_self.bin", shape);
    auto sa_ref = load_reference("test_data/reference/dec_l0_self_attn.bin", shape);
    auto after_sa_ref = load_reference("test_data/reference/dec_l0_after_sa.bin", shape);
    auto norm_xa_q_ref = load_reference("test_data/reference/dec_l0_norm_xa_q.bin", shape);
    auto xa_ref = load_reference("test_data/reference/dec_l0_cross_attn.bin", shape);
    auto after_xa_ref = load_reference("test_data/reference/dec_l0_after_xa.bin", shape);
    auto norm_ff_ref = load_reference("test_data/reference/dec_l0_norm_ff.bin", shape);
    auto ff_ref = load_reference("test_data/reference/dec_l0_ff_out.bin", shape);
    auto l0_out_ref = load_reference("test_data/reference/dec_l0_out.bin", shape);

    int64_t d_model = dec_input_shape[0];
    int64_t dec_seq = dec_input_shape[1];
    int64_t enc_seq = enc_out_shape[1];

    printf("d_model=%lld, dec_seq=%lld, enc_seq=%lld\n",
           (long long)d_model, (long long)dec_seq, (long long)enc_seq);

    // Build graph for step-by-step debugging
    size_t ctx_size = ggml_tensor_overhead() * 512 + 128 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensors
    struct ggml_tensor * dec_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, dec_seq);
    ggml_set_name(dec_input, "dec_input");
    ggml_set_input(dec_input);

    struct ggml_tensor * enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, enc_seq);
    ggml_set_name(enc_out, "enc_out");
    ggml_set_input(enc_out);

    // === Step 1: norm_self ===
    struct ggml_tensor * norm_self = magpie_build_layer_norm(ctx0, dec_input, layer.norm_self_w, hp.eps);
    ggml_set_name(norm_self, "norm_self");
    ggml_set_output(norm_self);

    // === Step 2: self-attention ===
    struct ggml_tensor * sa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dec_seq, dec_seq);
    ggml_set_name(sa_mask, "sa_mask");
    ggml_set_input(sa_mask);

    struct ggml_tensor * sa_out = magpie_build_self_attention_with_mask(
        ctx0, norm_self, layer.sa_qkv_w, layer.sa_out_w,
        hp.dec_sa_heads, true, sa_mask);
    ggml_set_name(sa_out, "sa_out");
    ggml_set_output(sa_out);

    // === Step 3: residual after SA ===
    struct ggml_tensor * after_sa = ggml_add(ctx0, sa_out, dec_input);
    ggml_set_name(after_sa, "after_sa");
    ggml_set_output(after_sa);

    // === Step 4: norm_xa_query ===
    struct ggml_tensor * norm_xa_q = magpie_build_layer_norm(ctx0, after_sa, layer.norm_xa_q_w, hp.eps);
    ggml_set_name(norm_xa_q, "norm_xa_q");
    ggml_set_output(norm_xa_q);

    // === Step 5: norm_xa_memory ===
    struct ggml_tensor * norm_xa_mem = magpie_build_layer_norm(ctx0, enc_out, layer.norm_xa_mem_w, hp.eps);
    ggml_set_name(norm_xa_mem, "norm_xa_mem");
    ggml_set_output(norm_xa_mem);

    // === Step 6: cross-attention ===
    struct ggml_tensor * xa_out = magpie_build_cross_attention(
        ctx0, norm_xa_q, norm_xa_mem,
        layer.xa_q_w, layer.xa_kv_w, layer.xa_out_w,
        hp.dec_xa_heads, hp.dec_xa_d_head);
    ggml_set_name(xa_out, "xa_out");
    ggml_set_output(xa_out);

    // === Step 7: residual after XA ===
    struct ggml_tensor * after_xa = ggml_add(ctx0, xa_out, after_sa);
    ggml_set_name(after_xa, "after_xa");
    ggml_set_output(after_xa);

    // === Step 8: norm_ff ===
    struct ggml_tensor * norm_ff = magpie_build_layer_norm(ctx0, after_xa, layer.norm_ff_w, hp.eps);
    ggml_set_name(norm_ff, "norm_ff");
    ggml_set_output(norm_ff);

    // === Step 9: FFN ===
    struct ggml_tensor * ff_out = magpie_build_conv_ffn(ctx0, norm_ff, layer.ff_proj_w, layer.ff_out_w, hp.dec_kernel);
    ggml_set_name(ff_out, "ff_out");
    ggml_set_output(ff_out);

    // === Step 10: final residual ===
    struct ggml_tensor * l0_out = ggml_add(ctx0, ff_out, after_xa);
    ggml_set_name(l0_out, "l0_out");
    ggml_set_output(l0_out);

    // Build and allocate graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(gf, l0_out);
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data
    ggml_backend_tensor_set(dec_input, dec_input_data.data(), 0, d_model * dec_seq * sizeof(float));
    ggml_backend_tensor_set(enc_out, enc_out_data.data(), 0, d_model * enc_seq * sizeof(float));

    // Set causal mask
    std::vector<float> mask_data(dec_seq * dec_seq);
    fill_causal_mask_f32(mask_data.data(), dec_seq);
    ggml_backend_tensor_set(sa_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

    printf("\nComputing...\n\n");
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get results and compare
    printf("=== Step-by-step Comparison ===\n\n");

    std::vector<float> result(d_model * dec_seq);

    // norm_self
    ggml_backend_tensor_get(norm_self, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("1. norm_self", result.data(), norm_self_ref.data(), result.size());

    // sa_out
    ggml_backend_tensor_get(sa_out, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("2. self_attn", result.data(), sa_ref.data(), result.size());

    // after_sa
    ggml_backend_tensor_get(after_sa, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("3. after_sa", result.data(), after_sa_ref.data(), result.size());

    // norm_xa_q
    ggml_backend_tensor_get(norm_xa_q, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("4. norm_xa_q", result.data(), norm_xa_q_ref.data(), result.size());

    // xa_out
    ggml_backend_tensor_get(xa_out, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("5. cross_attn", result.data(), xa_ref.data(), result.size());

    // after_xa
    ggml_backend_tensor_get(after_xa, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("6. after_xa", result.data(), after_xa_ref.data(), result.size());

    // norm_ff
    ggml_backend_tensor_get(norm_ff, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("7. norm_ff", result.data(), norm_ff_ref.data(), result.size());

    // ff_out
    ggml_backend_tensor_get(ff_out, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("8. ff_out", result.data(), ff_ref.data(), result.size());

    // l0_out
    ggml_backend_tensor_get(l0_out, result.data(), 0, result.size() * sizeof(float));
    compare_tensors("9. l0_out", result.data(), l0_out_ref.data(), result.size());

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    return 0;
}
