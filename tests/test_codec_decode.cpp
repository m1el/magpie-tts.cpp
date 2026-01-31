/**
 * Test codec decoder with continuous latent input (bypassing FSQ)
 *
 * This tests the HiFiGAN decoder directly with continuous latent values
 * loaded from PyTorch reference.
 */

#include "magpie.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>

// Load reference tensor from binary file (column-major format)
bool load_reference(const char * path, std::vector<float> & data, std::vector<int64_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return false;
    }

    // Read shape (4 x int64, GGML reversed order)
    int64_t dims[4];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));

    shape = {dims[0], dims[1], dims[2], dims[3]};

    // Calculate total elements
    size_t total = 1;
    for (int i = 0; i < 4; i++) {
        if (dims[i] > 0) total *= dims[i];
    }

    // Read data
    data.resize(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));

    return true;
}

float compute_max_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float max_diff = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int main(int argc, char ** argv) {
    const char * codec_path = "weights/nano-codec-f32.gguf";
    const char * ref_dir = "test_data/reference/codec";

    if (argc > 1) codec_path = argv[1];
    if (argc > 2) ref_dir = argv[2];

    printf("=== Test Codec Decoder ===\n");
    printf("Codec: %s\n", codec_path);
    printf("Reference: %s\n\n", ref_dir);

    // Load codec
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "ERROR: Failed to load codec\n");
        return 1;
    }
    printf("Codec loaded successfully\n\n");

    // Load reference latent (continuous values, bypassing FSQ)
    std::string latent_path = std::string(ref_dir) + "/codec_latent.bin";
    std::vector<float> latent_data;
    std::vector<int64_t> latent_shape;

    if (!load_reference(latent_path.c_str(), latent_data, latent_shape)) {
        fprintf(stderr, "ERROR: Could not load reference latent from %s\n", latent_path.c_str());
        fprintf(stderr, "Run: uv run scripts/inspect_codec.py --num-frames 5\n");
        magpie_codec_free(codec);
        return 1;
    }

    printf("Loaded reference latent: [%lld, %lld]\n",
           (long long)latent_shape[0], (long long)latent_shape[1]);

    // Reference was saved from PyTorch [B=1, C=32, T=5] with flatten('C')
    // GGML shape header has reversed dims: ne[0]=T=5, ne[1]=C=32
    // In C order, last dim (T) varies fastest, which matches GGML's ne[0]
    // Data can be loaded directly without transposition
    int n_frames = latent_shape[0];   // T=5 (ne[0])
    int latent_dim = latent_shape[1]; // C=32 (ne[1])

    printf("  n_frames (T): %d\n", n_frames);
    printf("  latent_dim (IC): %d\n", latent_dim);
    printf("  Data layout: time varies fastest (GGML ne[0]=T)\n");

    // Expected output size
    int expected_samples = n_frames * codec->hparams.hop_length;
    printf("  expected output: %d samples\n\n", expected_samples);

    // For now, test just the pre-conv layer to verify basic operations work
    printf("=== Testing Pre-Conv Layer ===\n");

    // Load reference pre-conv output
    std::string preconv_path = std::string(ref_dir) + "/codec_pre_conv.bin";
    std::vector<float> preconv_ref;
    std::vector<int64_t> preconv_shape;

    if (load_reference(preconv_path.c_str(), preconv_ref, preconv_shape)) {
        printf("Loaded pre-conv reference: [%lld, %lld]\n",
               (long long)preconv_shape[0], (long long)preconv_shape[1]);
    } else {
        printf("Pre-conv reference not found, skipping comparison\n");
    }

    // Build and run pre-conv graph
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "ERROR: Failed to init ggml context\n");
        magpie_codec_free(codec);
        return 1;
    }

    // Create input tensor for ggml_conv_1d
    // ggml_conv_1d expects input as [seq_length, in_channels] with ne[0]=T, ne[1]=IC
    // Reference data was saved in C order where T (ne[0]) varies fastest
    // This matches GGML expectations - no transposition needed
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, latent_dim);
    ggml_set_name(latent, "latent");
    ggml_set_input(latent);

    // Build pre-conv: [latent_dim, n_frames] -> [base_ch, n_frames]
    struct ggml_tensor * preconv_out = magpie_codec_build_causal_conv1d(
        ctx0, latent, codec->pre_conv_w, codec->pre_conv_b, 1);
    ggml_set_name(preconv_out, "preconv_out");
    ggml_set_output(preconv_out);

    // Create and allocate graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, preconv_out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(codec->backend));
    if (!ggml_gallocr_reserve(allocr, gf)) {
        fprintf(stderr, "ERROR: Failed to reserve graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        magpie_codec_free(codec);
        return 1;
    }

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "ERROR: Failed to alloc graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        magpie_codec_free(codec);
        return 1;
    }

    // Set input data (reference latent is already in column-major format)
    ggml_backend_tensor_set(latent, latent_data.data(), 0, latent_data.size() * sizeof(float));

    // Compute
    printf("Computing pre-conv...\n");
    if (ggml_backend_graph_compute(codec->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: Graph compute failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        magpie_codec_free(codec);
        return 1;
    }

    // Get output
    size_t out_size = ggml_nelements(preconv_out);
    std::vector<float> preconv_result(out_size);
    ggml_backend_tensor_get(preconv_out, preconv_result.data(), 0, out_size * sizeof(float));

    printf("Pre-conv output shape: [%lld, %lld]\n",
           (long long)preconv_out->ne[0], (long long)preconv_out->ne[1]);

    // Compare with reference
    bool preconv_pass = false;
    if (!preconv_ref.empty()) {
        float max_diff = compute_max_diff(preconv_result, preconv_ref);
        printf("Max diff vs reference: %.6f\n", max_diff);

        if (max_diff < 0.01f) {
            printf("[PASS] Pre-conv matches reference!\n");
            preconv_pass = true;
        } else {
            printf("[FAIL] Pre-conv mismatch!\n");
            printf("\nFirst 8 values:\n");
            printf("  GGML: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", preconv_result[i]);
            printf("\n  Ref:  ");
            for (int i = 0; i < 8; i++) printf("%.4f ", preconv_ref[i]);
            printf("\n");
        }
    }

    // Cleanup first test
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);

    // =========================================================================
    // Test HalfSnake Activation
    // =========================================================================
    printf("\n=== Testing HalfSnake Activation ===\n");

    // Load reference activation output
    std::string act_path = std::string(ref_dir) + "/codec_act_0.bin";
    std::vector<float> act_ref;
    std::vector<int64_t> act_shape;

    bool act_pass = false;
    if (load_reference(act_path.c_str(), act_ref, act_shape)) {
        printf("Loaded activation reference: [%lld, %lld]\n",
               (long long)act_shape[0], (long long)act_shape[1]);

        // Build graph: pre_conv output -> activation
        struct ggml_init_params params2 = {
            .mem_size   = 64 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context * ctx1 = ggml_init(params2);

        // Use pre_conv result as input
        int act_T = preconv_shape[0];
        int act_C = preconv_shape[1];
        struct ggml_tensor * preconv_in = ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, act_T, act_C);
        ggml_set_name(preconv_in, "preconv_in");
        ggml_set_input(preconv_in);

        // Apply HalfSnake activation
        struct ggml_tensor * act_out = magpie_codec_build_half_snake(
            ctx1, preconv_in, codec->upsample_layers[0].act_alpha);
        ggml_set_name(act_out, "act_out");
        ggml_set_output(act_out);

        // Create and run graph
        struct ggml_cgraph * gf1 = ggml_new_graph(ctx1);
        ggml_build_forward_expand(gf1, act_out);

        ggml_gallocr_t allocr1 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(codec->backend));
        ggml_gallocr_reserve(allocr1, gf1);
        ggml_gallocr_alloc_graph(allocr1, gf1);

        ggml_backend_tensor_set(preconv_in, preconv_result.data(), 0, preconv_result.size() * sizeof(float));

        printf("Computing HalfSnake activation...\n");
        ggml_backend_graph_compute(codec->backend, gf1);

        size_t act_size = ggml_nelements(act_out);
        std::vector<float> act_result(act_size);
        ggml_backend_tensor_get(act_out, act_result.data(), 0, act_size * sizeof(float));

        printf("Activation output shape: [%lld, %lld]\n",
               (long long)act_out->ne[0], (long long)act_out->ne[1]);

        float max_diff = compute_max_diff(act_result, act_ref);
        printf("Max diff vs reference: %.6f\n", max_diff);

        if (max_diff < 0.01f) {
            printf("[PASS] HalfSnake activation matches reference!\n");
            act_pass = true;
        } else {
            printf("[FAIL] HalfSnake activation mismatch!\n");
            printf("\nFirst 8 values (channel 0):\n");
            printf("  GGML: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", act_result[i]);
            printf("\n  Ref:  ");
            for (int i = 0; i < 8; i++) printf("%.4f ", act_ref[i]);
            printf("\n");

            // Find location of max diff
            int max_idx = 0;
            float max_d = 0.0f;
            for (size_t i = 0; i < std::min(act_result.size(), act_ref.size()); i++) {
                float d = fabsf(act_result[i] - act_ref[i]);
                if (d > max_d) { max_d = d; max_idx = i; }
            }
            int act_T = act_shape[0];
            int act_C = act_shape[1];
            int t_idx = max_idx % act_T;
            int c_idx = max_idx / act_T;
            printf("\nMax diff at flat_idx=%d (t=%d, c=%d):\n", max_idx, t_idx, c_idx);
            printf("  GGML: %.6f, Ref: %.6f, diff: %.6f\n",
                   act_result[max_idx], act_ref[max_idx], max_d);

            // Show some values around the boundary (channel 432 = start of second half)
            int boundary_start = 432 * act_T;
            printf("\nAround channel boundary (c=431-433):\n");
            for (int c = 431; c <= 433 && c < act_C; c++) {
                printf("  c=%d: GGML=%.4f, Ref=%.4f, diff=%.4f\n",
                       c, act_result[c * act_T], act_ref[c * act_T],
                       fabsf(act_result[c * act_T] - act_ref[c * act_T]));
            }

            // Check pre_conv input at the problematic location
            printf("\nPre-conv input at max_diff location (t=%d, c=%d):\n", t_idx, c_idx);
            int preconv_idx = t_idx + c_idx * act_T;
            printf("  GGML pre_conv: %.6f\n", preconv_result[preconv_idx]);
            printf("  Ref pre_conv:  %.6f\n", preconv_ref[preconv_idx]);
            printf("  Input diff: %.6f\n", fabsf(preconv_result[preconv_idx] - preconv_ref[preconv_idx]));
        }

        ggml_gallocr_free(allocr1);
        ggml_free(ctx1);
    } else {
        printf("Activation reference not found, skipping\n");
    }

    // =========================================================================
    // Test First Upsample (ConvTranspose1d) - SKIPPED
    // =========================================================================
    printf("\n=== Testing Upsample Layer 0 ===\n");
    printf("[SKIP] Depthwise ConvTranspose1d not yet implemented\n");
    printf("  GGML doesn't support grouped conv_transpose_1d.\n");
    printf("  Need to implement custom kernel or decomposition.\n");
    bool upsample_pass = false;  // Mark as not passed

    // Summary
    printf("\n=== Summary ===\n");
    printf("Pre-conv:        %s\n", preconv_pass ? "[PASS]" : "[FAIL]");
    printf("HalfSnake:       %s\n", act_pass ? "[PASS]" : "[FAIL]");
    printf("Upsample:        %s\n", upsample_pass ? "[PASS]" : "[FAIL]");

    magpie_codec_free(codec);

    printf("\nTest complete!\n");
    return (preconv_pass && act_pass && upsample_pass) ? 0 : 1;
}
