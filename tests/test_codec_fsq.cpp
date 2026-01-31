/**
 * Test FSQ (Finite Scalar Quantization) dequantization
 *
 * Compares GGML FSQ dequantization output against PyTorch reference.
 */

#include "magpie.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>

// Load reference tensor from binary file
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

// FSQ dequantization (CPU reference implementation)
void fsq_dequantize_cpu(
    const int32_t * codes,      // [num_cb, n_frames]
    float * latent,             // [n_frames, latent_dim] GGML layout (ne[0]=T, ne[1]=C)
    int num_cb,
    int n_frames)
{
    // FSQ parameters (matches nano-codec)
    const int dims_per_cb = 4;
    const int dim_base_index[4] = {1, 8, 56, 336};
    const int num_levels[4] = {8, 7, 6, 6};
    const int latent_dim = num_cb * dims_per_cb;  // 32

    for (int cb = 0; cb < num_cb; cb++) {
        for (int t = 0; t < n_frames; t++) {
            int index = codes[cb * n_frames + t];

            for (int d = 0; d < dims_per_cb; d++) {
                // nonneg = (index // dim_base_index[d]) % num_levels[d]
                int nonneg = (index / dim_base_index[d]) % num_levels[d];

                // code = (nonneg - num_levels[d]//2) / (num_levels[d]//2)
                int half_levels = num_levels[d] / 2;
                float code = (float)(nonneg - half_levels) / (float)half_levels;

                // Store in GGML layout: ne[0]=T varies fastest
                // Element at (t, c) is at index: t + c * n_frames
                int c = cb * dims_per_cb + d;
                int latent_idx = t + c * n_frames;
                latent[latent_idx] = code;
            }
        }
    }
}

int main(int argc, char ** argv) {
    const char * ref_dir = "test_data/reference/codec";
    if (argc > 1) {
        ref_dir = argv[1];
    }

    printf("=== Test FSQ Dequantization ===\n\n");

    // Test with known codes
    const int num_cb = 8;
    const int n_frames = 5;
    const int latent_dim = 32;

    // Use same random seed as Python script
    srand(42);

    // Generate test codes (same as Python)
    std::vector<int32_t> codes(num_cb * n_frames);
    // Load codes from reference file
    std::string codes_path = std::string(ref_dir) + "/codec_input_codes.bin";
    std::vector<float> codes_float;
    std::vector<int64_t> codes_shape;

    if (!load_reference(codes_path.c_str(), codes_float, codes_shape)) {
        printf("Could not load reference codes, using random codes\n");
        for (int i = 0; i < num_cb * n_frames; i++) {
            codes[i] = rand() % 2016;
        }
    } else {
        printf("Loaded codes from %s\n", codes_path.c_str());
        printf("  Shape: [%lld, %lld, %lld, %lld]\n",
               (long long)codes_shape[0], (long long)codes_shape[1],
               (long long)codes_shape[2], (long long)codes_shape[3]);
        // Convert float codes to int32 (they were stored as float in reference)
        for (size_t i = 0; i < codes_float.size(); i++) {
            codes[i] = (int32_t)codes_float[i];
        }
    }

    printf("\nTest codes (first 8): ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", codes[i]);
    }
    printf("\n\n");

    // Run FSQ dequantization
    std::vector<float> latent(latent_dim * n_frames);
    fsq_dequantize_cpu(codes.data(), latent.data(), num_cb, n_frames);

    printf("Dequantized latent (first 8 values): ");
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", latent[i]);
    }
    printf("\n\n");

    // Load reference latent if available
    std::string latent_path = std::string(ref_dir) + "/codec_latent.bin";
    std::vector<float> ref_latent;
    std::vector<int64_t> ref_shape;

    if (load_reference(latent_path.c_str(), ref_latent, ref_shape)) {
        printf("Loaded reference latent from %s\n", latent_path.c_str());
        printf("  Shape: [%lld, %lld, %lld, %lld]\n",
               (long long)ref_shape[0], (long long)ref_shape[1],
               (long long)ref_shape[2], (long long)ref_shape[3]);

        // Compare
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int n_diff = 0;

        for (size_t i = 0; i < ref_latent.size() && i < latent.size(); i++) {
            float diff = fabsf(latent[i] - ref_latent[i]);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
            if (diff > 0.001f) n_diff++;
        }

        printf("\n=== Comparison ===\n");
        printf("  Max diff: %.6f\n", max_diff);
        printf("  Avg diff: %.6f\n", sum_diff / ref_latent.size());
        printf("  Values with diff > 0.001: %d / %zu\n", n_diff, ref_latent.size());

        if (max_diff < 0.01f) {
            printf("\n[PASS] FSQ dequantization matches reference!\n");
        } else {
            printf("\n[FAIL] FSQ dequantization mismatch!\n");

            // Print first few values for debugging
            printf("\nFirst 8 values comparison:\n");
            printf("  GGML:      ");
            for (int i = 0; i < 8; i++) printf("%.4f ", latent[i]);
            printf("\n  Reference: ");
            for (int i = 0; i < 8; i++) printf("%.4f ", ref_latent[i]);
            printf("\n");
            return 1;
        }
    } else {
        printf("No reference latent file found at %s\n", latent_path.c_str());
        printf("Run: uv run scripts/inspect_codec.py --num-frames 5\n");
    }

    printf("\nTest complete!\n");
    return 0;
}
