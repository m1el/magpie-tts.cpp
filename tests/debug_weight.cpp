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

    // Load PyTorch attention output reference
    std::vector<float> ref_attn_out;
    int64_t attn_shape[4];
    read_ref("test_data/reference/debug_l1_attn_out.bin", ref_attn_out, attn_shape);

    // Get weight
    std::vector<float> weight(d_model * d_model);
    ggml_backend_tensor_get(layer->sa_out_w, weight.data(), 0, weight.size() * sizeof(float));

    fprintf(stderr, "Weight tensor shape: [%lld, %lld]\n",
            (long long)layer->sa_out_w->ne[0], (long long)layer->sa_out_w->ne[1]);
    fprintf(stderr, "Weight tensor nbytes: %zu\n", ggml_nbytes(layer->sa_out_w));

    // Compute output projection manually using GGML semantics:
    // Result[o, t] = sum_k W_ggml[k, o] * A_ggml[k, t]
    // W_ggml[k, o] at offset k + o * 768
    // A_ggml[k, t] at offset k + t * 768 = ref_attn_out[t * 768 + k] (same offset)

    int t = 9, o = 410;
    float sum = 0;
    for (int k = 0; k < d_model; k++) {
        // W_ggml[k, o] at memory offset k + o * 768
        float w = weight[k + o * d_model];
        // A at offset k + t * 768 = ref_attn_out[t * 768 + k] (if using PyTorch file)
        float a = ref_attn_out[t * d_model + k];
        sum += w * a;
    }
    fprintf(stderr, "\nGGML-style manual computation (using GGML indexing):\n");
    fprintf(stderr, "  Result[o=%d, t=%d] = %.6f (expected: 0.246909)\n", o, t, sum);

    // Also try PyTorch-style indexing:
    // Result[t, o] = sum_k A[t, k] * W[o, k]
    // W[o, k] at offset o * 768 + k (C row-major)
    // A[t, k] at offset t * 768 + k
    sum = 0;
    for (int k = 0; k < d_model; k++) {
        float w = weight[o * d_model + k];  // PyTorch W[o, k]
        float a = ref_attn_out[t * d_model + k];
        sum += w * a;
    }
    fprintf(stderr, "\nPyTorch-style manual computation (C row-major indexing):\n");
    fprintf(stderr, "  Result[t=%d, o=%d] = %.6f (expected: 0.246909)\n", t, o, sum);

    // The two should give the same result since they're computing the same thing
    // but with different index interpretations of the same memory

    // Now let's check: maybe the issue is that the weight in GGML has different values?
    // Let's dump a few weight values and compare to what debug_o_proj.cpp showed
    fprintf(stderr, "\nWeight samples:\n");
    fprintf(stderr, "  weight[0] = %.6f\n", weight[0]);
    fprintf(stderr, "  weight[1] = %.6f\n", weight[1]);
    fprintf(stderr, "  weight[768] = %.6f\n", weight[768]);
    fprintf(stderr, "  weight[410*768] = %.6f\n", weight[410*768]);
    fprintf(stderr, "  weight[410*768+1] = %.6f\n", weight[410*768+1]);

    magpie_free(ctx);
    return 0;
}
