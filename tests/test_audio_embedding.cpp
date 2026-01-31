#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// Read binary reference file (format: 4 x int64 shape + float32 data)
static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    // Read shape (4 x int64)
    if (fread(shape, sizeof(int64_t), 4, f) != 4) {
        fprintf(stderr, "Failed to read shape from: %s\n", path);
        fclose(f);
        return false;
    }

    // Calculate total elements
    int64_t n_elements = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n_elements);

    // Read data
    if (fread(data.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// Read codes reference file (format: 4 x int64 shape + float32 data representing codes)
static bool read_codes(const char * path, std::vector<int32_t> & codes, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    // Read shape (4 x int64)
    if (fread(shape, sizeof(int64_t), 4, f) != 4) {
        fprintf(stderr, "Failed to read shape from: %s\n", path);
        fclose(f);
        return false;
    }

    // Calculate total elements
    int64_t n_elements = shape[0] * shape[1] * shape[2] * shape[3];

    // Read as float32 (the dump script converts to float)
    std::vector<float> float_codes(n_elements);
    if (fread(float_codes.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path);
        fclose(f);
        return false;
    }

    // Convert to int32
    codes.resize(n_elements);
    for (size_t i = 0; i < (size_t)n_elements; i++) {
        codes[i] = (int32_t)float_codes[i];
    }

    fclose(f);
    return true;
}

// Compare tensors with tolerance
static bool compare_tensors(const float * a, const float * b, size_t n, float rtol, float atol) {
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(a[i] - b[i]);
        float rel = diff / (std::fabs(b[i]) + 1e-8f);

        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;

        if (diff > atol && rel > rtol) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                fprintf(stderr, "  Mismatch at %zu: got %.6f, expected %.6f (diff=%.6f, rel=%.4f)\n",
                        i, a[i], b[i], diff, rel);
            }
        }
    }

    fprintf(stderr, "  Max diff: %.6f, max rel: %.4f, mismatches: %zu/%zu\n",
            max_diff, max_rel, mismatch_count, n);

    return mismatch_count == 0;
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * codes_path = "test_data/reference/manual_audio_bos.bin";
    const char * emb_path = "test_data/reference/manual_audio_emb.bin";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) codes_path = argv[2];
    if (argc > 3) emb_path = argv[3];

    // Load model
    fprintf(stderr, "Loading model from: %s\n", model_path);
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Check audio embedding weights are loaded
    for (int cb = 0; cb < 8; cb++) {
        if (!ctx->model.embeddings.audio_emb_w[cb]) {
            fprintf(stderr, "audio_emb_w[%d] not loaded!\n", cb);
            magpie_free(ctx);
            return 1;
        }
    }

    ggml_tensor * emb_w0 = ctx->model.embeddings.audio_emb_w[0];
    fprintf(stderr, "audio_emb_w[0] shape: [%lld, %lld]\n",
            (long long)emb_w0->ne[0], (long long)emb_w0->ne[1]);

    // Read reference codes
    std::vector<int32_t> codes;
    int64_t codes_shape[4];
    if (!read_codes(codes_path, codes, codes_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Loaded codes from %s\n", codes_path);
    fprintf(stderr, "Codes shape: [%lld, %lld, %lld, %lld]\n",
            (long long)codes_shape[0], (long long)codes_shape[1],
            (long long)codes_shape[2], (long long)codes_shape[3]);

    // Print codes
    fprintf(stderr, "Codes: ");
    for (size_t i = 0; i < codes.size() && i < 16; i++) {
        fprintf(stderr, "%d ", codes[i]);
    }
    fprintf(stderr, "\n");

    // Read reference embeddings
    std::vector<float> expected;
    int64_t emb_shape[4];
    if (!read_reference(emb_path, expected, emb_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Reference embedding shape: [%lld, %lld, %lld, %lld]\n",
            (long long)emb_shape[0], (long long)emb_shape[1],
            (long long)emb_shape[2], (long long)emb_shape[3]);
    fprintf(stderr, "Reference embedding size: %zu elements\n", expected.size());

    // The codes from manual_audio_bos.bin are [1, 8, 1] in PyTorch = [batch, codebooks, seq]
    // In the binary file, shape is reversed: [1, 8, 1, 1] (padded)
    // The actual codes are 8 values (one per codebook)
    // For our GGML function, we need codes as [8] (just the 8 codebook values)

    // Extract the 8 codes (ignoring batch dimension)
    // Shape [1, 8, 1, 1] means 8 elements total, one per codebook
    int n_codes = 8;
    std::vector<int32_t> flat_codes(n_codes);
    for (int i = 0; i < n_codes; i++) {
        flat_codes[i] = codes[i];  // Should be 2016 (BOS) for all
    }

    fprintf(stderr, "Flat codes for GGML: ");
    for (int i = 0; i < n_codes; i++) {
        fprintf(stderr, "%d ", flat_codes[i]);
    }
    fprintf(stderr, "\n");

    const int d_model = ctx->model.hparams.d_model;

    // Build computation graph
    size_t ctx_size = ggml_tensor_overhead() * 20 + 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to create compute context\n");
        magpie_free(ctx);
        return 1;
    }

    // Create codes tensor [8] - one code per codebook
    struct ggml_tensor * codes_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_codes);
    ggml_set_name(codes_tensor, "codes");
    ggml_set_input(codes_tensor);

    // Build audio embedding graph
    struct ggml_tensor * embedded = magpie_build_audio_embedding(
        ctx0, codes_tensor, &ctx->model.embeddings);

    if (!embedded) {
        fprintf(stderr, "magpie_build_audio_embedding returned null!\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    ggml_set_name(embedded, "audio_embedded");
    ggml_set_output(embedded);

    fprintf(stderr, "Embedded tensor shape: [%lld, %lld]\n",
            (long long)embedded->ne[0], (long long)embedded->ne[1]);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, embedded);

    // Allocate compute buffers
    if (!ggml_gallocr_reserve(ctx->state.allocr, gf)) {
        fprintf(stderr, "Failed to reserve allocator\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    if (!ggml_gallocr_alloc_graph(ctx->state.allocr, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    // Copy codes to device
    ggml_backend_tensor_set(codes_tensor, flat_codes.data(), 0, n_codes * sizeof(int32_t));

    // Run computation
    fprintf(stderr, "Running computation...\n");
    if (ggml_backend_graph_compute(ctx->model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph computation failed\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    // Get results - shape should be [d_model, 1] or [d_model]
    int64_t result_size = embedded->ne[0] * embedded->ne[1];
    std::vector<float> result(result_size);
    ggml_backend_tensor_get(embedded, result.data(), 0, result_size * sizeof(float));

    // Print first few values
    fprintf(stderr, "First 8 values: ");
    for (int i = 0; i < 8 && i < (int)result.size(); i++) {
        fprintf(stderr, "%.4f ", result[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Expected first 8: ");
    for (int i = 0; i < 8 && i < (int)expected.size(); i++) {
        fprintf(stderr, "%.4f ", expected[i]);
    }
    fprintf(stderr, "\n");

    // Compare - expected might have extra batch/seq dimensions, take first d_model values
    size_t compare_size = std::min(result.size(), expected.size());
    compare_size = std::min(compare_size, (size_t)d_model);

    fprintf(stderr, "\nComparing first %zu values:\n", compare_size);
    bool match = compare_tensors(result.data(), expected.data(), compare_size, 1e-4f, 1e-5f);

    ggml_free(ctx0);
    magpie_free(ctx);

    if (match) {
        fprintf(stderr, "\nSUCCESS: Audio embedding matches reference!\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAILURE: Audio embedding does not match reference\n");
        return 1;
    }
}
