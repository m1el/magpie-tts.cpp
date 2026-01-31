// Debug decoder input data
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

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) return 1;

    const auto & hp = mctx->model.hparams;

    // Load decoder input and norm_self reference
    int64_t dec_input_shape[4], norm_ref_shape[4];
    auto dec_input_data = load_reference("test_data/reference/dec_input.bin", dec_input_shape);
    auto norm_ref = load_reference("test_data/reference/dec_l0_norm_self.bin", norm_ref_shape);

    printf("dec_input shape: [%lld, %lld, %lld, %lld]\n",
           (long long)dec_input_shape[0], (long long)dec_input_shape[1],
           (long long)dec_input_shape[2], (long long)dec_input_shape[3]);
    printf("norm_ref shape: [%lld, %lld, %lld, %lld]\n",
           (long long)norm_ref_shape[0], (long long)norm_ref_shape[1],
           (long long)norm_ref_shape[2], (long long)norm_ref_shape[3]);

    // Print first 5 values of decoder input
    printf("\nDecoder input first5: %.6f %.6f %.6f %.6f %.6f\n",
           dec_input_data[0], dec_input_data[1], dec_input_data[2],
           dec_input_data[3], dec_input_data[4]);

    printf("Norm reference first5: %.6f %.6f %.6f %.6f %.6f\n",
           norm_ref[0], norm_ref[1], norm_ref[2], norm_ref[3], norm_ref[4]);

    // Do manual layer norm computation
    int64_t d_model = dec_input_shape[0];
    int64_t seq_len = dec_input_shape[1];

    // Get decoder layer 0 norm_self weight
    auto & layer = mctx->model.decoder.layers[0];
    std::vector<float> norm_w(d_model);
    ggml_backend_tensor_get(layer.norm_self_w, norm_w.data(), 0, d_model * sizeof(float));

    printf("norm_self_w first5: %.6f %.6f %.6f %.6f %.6f\n",
           norm_w[0], norm_w[1], norm_w[2], norm_w[3], norm_w[4]);

    // Manually compute layer norm on first position
    // LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * weight
    double sum = 0, sq_sum = 0;
    for (int64_t i = 0; i < d_model; i++) {
        sum += dec_input_data[i];  // First column
    }
    double mean = sum / d_model;

    for (int64_t i = 0; i < d_model; i++) {
        double diff = dec_input_data[i] - mean;
        sq_sum += diff * diff;
    }
    double var = sq_sum / d_model;
    double std = sqrt(var + hp.eps);

    printf("\nManual LayerNorm on position 0:\n");
    printf("  mean=%.6f, var=%.6f, std=%.6f\n", mean, var, std);

    printf("  Manual first5: ");
    for (int i = 0; i < 5; i++) {
        double normalized = (dec_input_data[i] - mean) / std;
        double result = normalized * norm_w[i];
        printf("%.6f ", result);
    }
    printf("\n");

    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           norm_ref[0], norm_ref[1], norm_ref[2], norm_ref[3], norm_ref[4]);

    magpie_free(mctx);
    return 0;
}
