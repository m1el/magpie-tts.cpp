// Test full decoder (12 layers) against PyTorch reference
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

// Build full decoder: 12 layers + final norm
struct ggml_tensor * magpie_build_full_decoder(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * encoder_out,
    struct magpie_decoder * decoder,
    const magpie_hparams * hparams) {

    if (!input || !encoder_out || !decoder || !hparams) return nullptr;

    const int64_t dec_seq = input->ne[1];

    // Create shared causal mask for all layers
    struct ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec_seq, dec_seq);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    struct ggml_tensor * x = input;

    // Process through all decoder layers
    for (int l = 0; l < hparams->dec_layers; l++) {
        auto & layer = decoder->layers[l];

        // Self-attention block (causal)
        struct ggml_tensor * residual = x;
        struct ggml_tensor * norm_self = magpie_build_layer_norm(ctx, x, layer.norm_self_w, hparams->eps);
        x = magpie_build_self_attention_with_mask(ctx, norm_self, layer.sa_qkv_w, layer.sa_out_w,
                                                   hparams->dec_sa_heads, true, causal_mask);
        x = ggml_add(ctx, x, residual);

        // Cross-attention block
        residual = x;
        struct ggml_tensor * norm_xa_q = magpie_build_layer_norm(ctx, x, layer.norm_xa_q_w, hparams->eps);
        struct ggml_tensor * norm_xa_mem = magpie_build_layer_norm(ctx, encoder_out, layer.norm_xa_mem_w, hparams->eps);
        x = magpie_build_cross_attention(ctx, norm_xa_q, norm_xa_mem,
                                          layer.xa_q_w, layer.xa_kv_w, layer.xa_out_w,
                                          hparams->dec_xa_heads, hparams->dec_xa_d_head);
        x = ggml_add(ctx, x, residual);

        // FFN block (kernel=1)
        residual = x;
        struct ggml_tensor * norm_ff = magpie_build_layer_norm(ctx, x, layer.norm_ff_w, hparams->eps);
        x = magpie_build_conv_ffn(ctx, norm_ff, layer.ff_proj_w, layer.ff_out_w, hparams->dec_kernel);
        x = ggml_add(ctx, x, residual);
    }

    // Final norm
    x = magpie_build_layer_norm(ctx, x, decoder->norm_out_w, hparams->eps);

    return x;
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
    int64_t dec_input_shape[4], enc_out_shape[4], dec_out_shape[4];
    auto dec_input_data = load_reference("test_data/reference/dec_input.bin", dec_input_shape);
    auto enc_out_data = load_reference("test_data/reference/dec_encoder_output.bin", enc_out_shape);
    auto dec_out_expected = load_reference("test_data/reference/dec_output.bin", dec_out_shape);

    if (dec_input_data.empty() || enc_out_data.empty() || dec_out_expected.empty()) {
        fprintf(stderr, "Failed to load reference data. Run: uv run scripts/dump_decoder_reference.py\n");
        magpie_free(mctx);
        return 1;
    }

    int64_t d_model = dec_input_shape[0];
    int64_t dec_seq = dec_input_shape[1];
    int64_t enc_seq = enc_out_shape[1];

    printf("d_model=%lld, dec_seq=%lld, enc_seq=%lld, dec_layers=%d\n",
           (long long)d_model, (long long)dec_seq, (long long)enc_seq, hp.dec_layers);

    // Build graph
    size_t ctx_size = ggml_tensor_overhead() * 2048 + 256 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensors
    struct ggml_tensor * dec_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, dec_seq);
    ggml_set_name(dec_input, "dec_input");
    ggml_set_input(dec_input);

    struct ggml_tensor * enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, enc_seq);
    ggml_set_name(enc_out, "enc_out");
    ggml_set_input(enc_out);

    // Build full decoder graph
    struct ggml_tensor * output = magpie_build_full_decoder(
        ctx0, dec_input, enc_out, &mctx->model.decoder, &hp);

    if (!output) {
        fprintf(stderr, "Failed to build decoder\n");
        ggml_free(ctx0);
        magpie_free(mctx);
        return 1;
    }

    ggml_set_name(output, "output");
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, output);
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate and run
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data
    ggml_backend_tensor_set(dec_input, dec_input_data.data(), 0, d_model * dec_seq * sizeof(float));
    ggml_backend_tensor_set(enc_out, enc_out_data.data(), 0, d_model * enc_seq * sizeof(float));

    // Set causal mask
    struct ggml_tensor * mask = ggml_get_tensor(ctx0, "causal_mask");
    if (mask) {
        printf("Setting causal mask [%lld, %lld]\n", (long long)mask->ne[0], (long long)mask->ne[1]);
        std::vector<float> mask_data(dec_seq * dec_seq);
        fill_causal_mask_f32(mask_data.data(), dec_seq);
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    printf("Computing full decoder (12 layers)...\n");
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get result
    std::vector<float> result(d_model * dec_seq);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Compare
    float max_diff = 0;
    int max_idx = 0;
    float sum_diff = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = fabsf(result[i] - dec_out_expected[i]);
        sum_diff += diff;
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }

    printf("\n=== Full Decoder Comparison (12 layers + final norm) ===\n");
    printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
           result[0], result[1], result[2], result[3], result[4]);
    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           dec_out_expected[0], dec_out_expected[1], dec_out_expected[2],
           dec_out_expected[3], dec_out_expected[4]);
    printf("  Max diff: %.6f at idx %d\n", max_diff, max_idx);
    printf("  Avg diff: %.6f\n", sum_diff / result.size());

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_diff < 0.1f) {
        printf("\nSUCCESS: Full decoder matches within tolerance!\n");
        return 0;
    } else {
        printf("\nFAILURE: Full decoder has large errors\n");
        return 1;
    }
}
