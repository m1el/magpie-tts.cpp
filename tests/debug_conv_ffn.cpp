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

static void compare_at(const char * name, const float * a, const float * b, size_t idx) {
    float diff = std::fabs(a[idx] - b[idx]);
    fprintf(stderr, "  %s[%zu]: got=%.6f, exp=%.6f, diff=%.6f\n", name, idx, a[idx], b[idx], diff);
}

int main() {
    magpie_context * ctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!ctx) return 1;

    const int d_model = ctx->model.hparams.d_model;
    const int d_ffn = ctx->model.hparams.d_ffn;
    const int kernel_size = ctx->model.hparams.enc_kernel;

    // Get layer 1 weights
    magpie_encoder_layer * layer = &ctx->model.encoder.layers[1];

    fprintf(stderr, "d_model=%d, d_ffn=%d, kernel_size=%d\n", d_model, d_ffn, kernel_size);
    fprintf(stderr, "ff_proj_w shape: [%lld, %lld, %lld]\n",
            (long long)layer->ff_proj_w->ne[0], (long long)layer->ff_proj_w->ne[1],
            (long long)layer->ff_proj_w->ne[2]);
    fprintf(stderr, "ff_out_w shape: [%lld, %lld, %lld]\n",
            (long long)layer->ff_out_w->ne[0], (long long)layer->ff_out_w->ne[1],
            (long long)layer->ff_out_w->ne[2]);

    // Load reference data - layer0 output and layer1 intermediate
    std::vector<float> layer0_out;
    int64_t shape[4];
    read_reference("test_data/reference/manual_enc_layer0_out.bin", layer0_out, shape);
    const int seq_len = (int)shape[2];

    // Check if we have FFN debug reference data
    std::vector<float> debug_ffn_input, debug_ffn_proj_out, debug_ffn_gelu, debug_ffn_output;
    int64_t ffn_shape[4];

    bool has_ffn_debug = read_reference("test_data/reference/debug_ffn_input.bin", debug_ffn_input, ffn_shape);
    if (has_ffn_debug) {
        fprintf(stderr, "\nFound FFN debug data!\n");
        read_reference("test_data/reference/debug_ffn_proj_out.bin", debug_ffn_proj_out, ffn_shape);
        read_reference("test_data/reference/debug_ffn_gelu.bin", debug_ffn_gelu, ffn_shape);
        read_reference("test_data/reference/debug_ffn_output.bin", debug_ffn_output, ffn_shape);
    }

    // Let's manually check the weight values at specific positions
    fprintf(stderr, "\n=== Checking weight values ===\n");

    // Read some weights to verify they loaded correctly
    std::vector<float> proj_w_data(layer->ff_proj_w->ne[0] * layer->ff_proj_w->ne[1] * layer->ff_proj_w->ne[2]);
    ggml_backend_tensor_get(layer->ff_proj_w, proj_w_data.data(), 0, proj_w_data.size() * sizeof(float));

    fprintf(stderr, "ff_proj_w first 5 values: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", proj_w_data[i]);
    fprintf(stderr, "\n");

    // Check norm_ff_w
    std::vector<float> norm_w_data(d_model);
    ggml_backend_tensor_get(layer->norm_ff_w, norm_w_data.data(), 0, norm_w_data.size() * sizeof(float));
    fprintf(stderr, "norm_ff_w first 5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", norm_w_data[i]);
    fprintf(stderr, "\n");

    // Now let's test with a simple known input
    fprintf(stderr, "\n=== Manual FFN test with layer 1 ===\n");

    size_t ctx_size = ggml_tensor_overhead() * 200 + 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Use a simple input - first we need to get the normalized input to FFN
    // which is norm2(res1) = norm_ff(input + self_attn(norm_self(input)))
    // For simplicity, let's just test the FFN with a simple constant input

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    // Build FFN manually to inspect each stage
    int64_t pad_left = kernel_size - 1;

    // Pad input
    struct ggml_tensor * padded = ggml_pad_ext(ctx0, input, 0, 0, pad_left, 0, 0, 0, 0, 0);
    ggml_set_name(padded, "padded"); ggml_set_output(padded);

    // Permute proj weight
    struct ggml_tensor * proj_perm = ggml_cont(ctx0, ggml_permute(ctx0, layer->ff_proj_w, 2, 0, 1, 3));
    ggml_set_name(proj_perm, "proj_perm"); ggml_set_output(proj_perm);

    // Do the convolution as sum of matmuls
    struct ggml_tensor * term0, * term1, * term2;

    struct ggml_tensor * input_k0 = ggml_view_2d(ctx0, padded, d_model, seq_len, padded->nb[1], 0);
    struct ggml_tensor * w_k0 = ggml_view_2d(ctx0, proj_perm, d_model, d_ffn, proj_perm->nb[1], 0);
    term0 = ggml_mul_mat(ctx0, w_k0, input_k0);
    ggml_set_name(term0, "term0"); ggml_set_output(term0);

    struct ggml_tensor * input_k1 = ggml_view_2d(ctx0, padded, d_model, seq_len, padded->nb[1], padded->nb[1]);
    struct ggml_tensor * w_k1 = ggml_view_2d(ctx0, proj_perm, d_model, d_ffn, proj_perm->nb[1], proj_perm->nb[2]);
    term1 = ggml_mul_mat(ctx0, w_k1, input_k1);
    ggml_set_name(term1, "term1"); ggml_set_output(term1);

    struct ggml_tensor * input_k2 = ggml_view_2d(ctx0, padded, d_model, seq_len, padded->nb[1], 2 * padded->nb[1]);
    struct ggml_tensor * w_k2 = ggml_view_2d(ctx0, proj_perm, d_model, d_ffn, proj_perm->nb[1], 2 * proj_perm->nb[2]);
    term2 = ggml_mul_mat(ctx0, w_k2, input_k2);
    ggml_set_name(term2, "term2"); ggml_set_output(term2);

    struct ggml_tensor * proj_out = ggml_add(ctx0, ggml_add(ctx0, term0, term1), term2);
    ggml_set_name(proj_out, "proj_out"); ggml_set_output(proj_out);

    struct ggml_tensor * gelu_out = ggml_gelu(ctx0, proj_out);
    ggml_set_name(gelu_out, "gelu_out"); ggml_set_output(gelu_out);

    // Full FFN output
    struct ggml_tensor * ffn_out = magpie_build_conv_ffn(ctx0, input, layer->ff_proj_w, layer->ff_out_w, kernel_size);
    ggml_set_name(ffn_out, "ffn_out"); ggml_set_output(ffn_out);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, ffn_out);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    // Use debug_ffn_input if available, otherwise use ones
    if (has_ffn_debug) {
        fprintf(stderr, "Using debug_ffn_input as input\n");
        ggml_backend_tensor_set(input, debug_ffn_input.data(), 0, debug_ffn_input.size() * sizeof(float));
    } else {
        std::vector<float> test_input(d_model * seq_len, 1.0f);
        ggml_backend_tensor_set(input, test_input.data(), 0, test_input.size() * sizeof(float));
    }

    ggml_backend_graph_compute(ctx->model.backend, gf);

    // Get results
    std::vector<float> buf_proj(d_ffn * seq_len), buf_gelu(d_ffn * seq_len), buf_ffn(d_model * seq_len);

    ggml_backend_tensor_get(proj_out, buf_proj.data(), 0, buf_proj.size() * sizeof(float));
    ggml_backend_tensor_get(gelu_out, buf_gelu.data(), 0, buf_gelu.size() * sizeof(float));
    ggml_backend_tensor_get(ffn_out, buf_ffn.data(), 0, buf_ffn.size() * sizeof(float));

    fprintf(stderr, "\nproj_out first 5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", buf_proj[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "gelu_out first 5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", buf_gelu[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "ffn_out first 5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", buf_ffn[i]);
    fprintf(stderr, "\n");

    if (has_ffn_debug) {
        fprintf(stderr, "\n=== Comparing to PyTorch reference ===\n");
        fprintf(stderr, "debug_ffn_proj_out first 5: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", debug_ffn_proj_out[i]);
        fprintf(stderr, "\n");

        fprintf(stderr, "debug_ffn_gelu first 5: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", debug_ffn_gelu[i]);
        fprintf(stderr, "\n");

        fprintf(stderr, "debug_ffn_output first 5: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", debug_ffn_output[i]);
        fprintf(stderr, "\n");

        // Find max diff
        float max_diff = 0;
        size_t max_idx = 0;
        for (size_t i = 0; i < buf_ffn.size(); i++) {
            float diff = std::fabs(buf_ffn[i] - debug_ffn_output[i]);
            if (diff > max_diff) { max_diff = diff; max_idx = i; }
        }
        fprintf(stderr, "\nFFN output max_diff=%.6f at idx %zu\n", max_diff, max_idx);
        compare_at("ffn_out", buf_ffn.data(), debug_ffn_output.data(), max_idx);
    }

    ggml_free(ctx0);
    magpie_free(ctx);
    return 0;
}
