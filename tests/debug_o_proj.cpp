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

int main() {
    magpie_context * ctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!ctx) return 1;

    const int d_model = ctx->model.hparams.d_model;
    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    // Load reference attention output (before o_net)
    std::vector<float> ref_attn_out;
    int64_t shape[4];
    if (!read_ref("test_data/reference/debug_l1_attn_out.bin", ref_attn_out, shape)) {
        fprintf(stderr, "Failed to load attention output reference\n");
        return 1;
    }
    int seq_len = (int)shape[2];
    fprintf(stderr, "Loaded ref attn_out: shape=[%lld, %lld, %lld, %lld]\n",
            (long long)shape[0], (long long)shape[1], (long long)shape[2], (long long)shape[3]);

    // Get o_net weight
    std::vector<float> weight_data(d_model * d_model);
    ggml_backend_tensor_get(layer->sa_out_w, weight_data.data(), 0, weight_data.size() * sizeof(float));
    fprintf(stderr, "sa_out_w shape: [%lld, %lld]\n",
            (long long)layer->sa_out_w->ne[0], (long long)layer->sa_out_w->ne[1]);

    // In PyTorch: sa_out[t, o] = sum_k attn_out[t, k] * weight[o, k]
    // weight is [768, 768] in PyTorch (out, in)

    // In GGML col-major storage:
    // weight_data is stored as C-order [768, 768] from numpy
    // GGML interprets this as col-major, so:
    // weight_ggml[i, j] = weight_data[j * 768 + i] = weight_numpy[j, i]
    // So weight_ggml[k, o] = weight_pytorch[o, k]

    // For position 7322 = (t=9, d=410):
    int t = 9, d = 410;
    int pos = d + t * d_model;  // GGML col-major position

    fprintf(stderr, "\nManual computation for sa_out[t=%d, d=%d]:\n", t, d);
    fprintf(stderr, "ref_attn_out[%d] = %.6f\n", pos, ref_attn_out[pos]);

    // Compute manually: sa_out[d, t] = sum_k weight_ggml[k, d] * attn_out[k, t]
    // In C-order storage: weight_ggml[k, d] = weight_data[d * 768 + k]
    //                    attn_out_ggml[k, t] = ref_attn_out[t * 768 + k] (assuming row-major file)

    // Wait - need to figure out the file format
    // From dump_attn_out.py, the file is written as:
    // write_tensor(attn_out_reshaped[0], ...) where attn_out_reshaped is [seq, d_model]
    // write_tensor writes in C row-major order

    // So ref_attn_out[t * d_model + k] = attn_out_pytorch[t, k]

    fprintf(stderr, "\nFirst 5 values of ref_attn_out for token 9:\n");
    for (int k = 0; k < 5; k++) {
        fprintf(stderr, "  attn_out[t=9, k=%d] = %.6f\n", k, ref_attn_out[9 * d_model + k]);
    }

    // Now compute sa_out[t=9, d=410] manually using PyTorch convention:
    // sa_out[t, o] = sum_k attn_out[t, k] * weight[o, k]
    float sum = 0;
    for (int k = 0; k < d_model; k++) {
        // weight[o=410, k] is at weight_data[410 * 768 + k] (C row-major from PyTorch)
        float w = weight_data[410 * d_model + k];
        // attn_out[t=9, k] is at ref_attn_out[9 * 768 + k]
        float a = ref_attn_out[9 * d_model + k];
        sum += w * a;
    }
    fprintf(stderr, "\nManual PyTorch-style computation:\n");
    fprintf(stderr, "  sa_out[t=9, d=410] = %.6f (expected 0.246909)\n", sum);

    // Now let's test what GGML computes
    // In GGML: result[o, t] = sum_k weight_ggml[k, o] * input_ggml[k, t]
    // weight_ggml[k, o] = weight_data[o * 768 + k] (since C-order is interpreted as col-major)
    // input_ggml[k, t] = ... depends on how we set it up

    // Let's also compute what GGML would produce if we fed it attn_out in col-major order:
    // If input_ggml[k, t] = attn_out_pytorch[t, k] (which is ref_attn_out[t * 768 + k]),
    // then we need to set up GGML input as:
    // ggml_input[k + t * 768] = ref_attn_out[t * 768 + k]
    // This means we DON'T transpose the input! Just use ref_attn_out directly.

    fprintf(stderr, "\n=== GGML matmul test ===\n");

    // Allocate GGML context
    size_t ctx_size = ggml_tensor_overhead() * 100 + 16 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * output = ggml_mul_mat(ctx0, layer->sa_out_w, input);
    ggml_set_name(output, "output"); ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 1024, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    // Set input - try different arrangements
    // Arrangement 1: Use ref_attn_out directly (C row-major to GGML col-major)
    // This should work if GGML expects attn_out in [d_model, seq] col-major
    // where element [k, t] = ref_attn_out[t * 768 + k]

    // Actually, for GGML col-major [d_model, seq]:
    // element [k, t] is at memory offset k + t * d_model
    // So we need: input_memory[k + t * d_model] = attn_out_pytorch[t, k] = ref_attn_out[t * d_model + k]
    // This is NOT the same memory layout! We need to transpose.

    std::vector<float> input_ggml(d_model * seq_len);
    for (int tt = 0; tt < seq_len; tt++) {
        for (int kk = 0; kk < d_model; kk++) {
            // GGML position [kk, tt] = kk + tt * d_model
            // PyTorch value at [tt, kk] = ref_attn_out[tt * d_model + kk]
            input_ggml[kk + tt * d_model] = ref_attn_out[tt * d_model + kk];
        }
    }

    ggml_backend_tensor_set(input, input_ggml.data(), 0, input_ggml.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->model.backend, gf);

    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // GGML result [d, t] is at position d + t * d_model
    int result_pos = d + t * d_model;
    fprintf(stderr, "GGML result[d=%d, t=%d] at offset %d = %.6f\n", d, t, result_pos, result[result_pos]);

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
