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

    // Read shape (4 x int64, reversed order in file)
    int64_t file_shape[4];
    if (fread(file_shape, sizeof(int64_t), 4, f) != 4) {
        fprintf(stderr, "Failed to read shape from: %s\n", path);
        fclose(f);
        return false;
    }

    // GGML order: ne[0], ne[1], ne[2], ne[3] (innermost to outermost)
    // File has them in reversed PyTorch order
    shape[0] = file_shape[0];  // innermost (d_model)
    shape[1] = file_shape[1];  // seq
    shape[2] = file_shape[2];  // batch
    shape[3] = file_shape[3];  // 1

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

// Read token reference file (format: 4 x int64 shape + float32 data representing tokens)
static bool read_tokens(const char * path, std::vector<int32_t> & tokens, int64_t shape[4]) {
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
    std::vector<float> float_tokens(n_elements);
    if (fread(float_tokens.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path);
        fclose(f);
        return false;
    }

    // Convert to int32
    tokens.resize(n_elements);
    for (size_t i = 0; i < (size_t)n_elements; i++) {
        tokens[i] = (int32_t)float_tokens[i];
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
    const char * tokens_path = "test_data/reference/manual_text_tokens.bin";
    const char * embedded_path = "test_data/reference/manual_text_embedded.bin";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) tokens_path = argv[2];
    if (argc > 3) embedded_path = argv[3];

    // Load model
    fprintf(stderr, "Loading model from: %s\n", model_path);
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Check text embedding weights are loaded
    if (!ctx->model.embeddings.text_emb_w) {
        fprintf(stderr, "text_emb_w not loaded!\n");
        magpie_free(ctx);
        return 1;
    }

    ggml_tensor * emb_w = ctx->model.embeddings.text_emb_w;
    fprintf(stderr, "text_emb_w shape: [%lld, %lld]\n",
            (long long)emb_w->ne[0], (long long)emb_w->ne[1]);

    // Read reference tokens
    std::vector<int32_t> tokens;
    int64_t token_shape[4];
    if (!read_tokens(tokens_path, tokens, token_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Loaded %zu tokens from %s\n", tokens.size(), tokens_path);
    fprintf(stderr, "Token shape: [%lld, %lld, %lld, %lld]\n",
            (long long)token_shape[0], (long long)token_shape[1],
            (long long)token_shape[2], (long long)token_shape[3]);

    // Print first few tokens
    fprintf(stderr, "Tokens: ");
    for (size_t i = 0; i < tokens.size() && i < 16; i++) {
        fprintf(stderr, "%d ", tokens[i]);
    }
    fprintf(stderr, "\n");

    // Read reference embeddings
    std::vector<float> expected;
    int64_t emb_shape[4];
    if (!read_reference(embedded_path, expected, emb_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Reference embedding shape: [%lld, %lld, %lld, %lld]\n",
            (long long)emb_shape[0], (long long)emb_shape[1],
            (long long)emb_shape[2], (long long)emb_shape[3]);

    // Build computation graph
    const int n_tokens = (int)tokens.size();
    const int d_model = ctx->model.hparams.d_model;

    // Create graph context
    size_t ctx_size = ggml_tensor_overhead() * 10 + 1024 * 1024;
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

    // Create token tensor
    struct ggml_tensor * token_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(token_tensor, "tokens");
    ggml_set_input(token_tensor);

    // Build embedding graph
    struct ggml_tensor * embedded = magpie_build_text_embedding(
        ctx0, token_tensor, &ctx->model.embeddings);

    if (!embedded) {
        fprintf(stderr, "magpie_build_text_embedding returned null!\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    ggml_set_name(embedded, "embedded");
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

    // Copy tokens to device
    ggml_backend_tensor_set(token_tensor, tokens.data(), 0, n_tokens * sizeof(int32_t));

    // Run computation
    fprintf(stderr, "Running computation...\n");
    if (ggml_backend_graph_compute(ctx->model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph computation failed\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    // Get results
    std::vector<float> result(n_tokens * d_model);
    ggml_backend_tensor_get(embedded, result.data(), 0, result.size() * sizeof(float));

    // Print first few values
    fprintf(stderr, "First 8 values: ");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f ", result[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Expected first 8: ");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f ", expected[i]);
    }
    fprintf(stderr, "\n");

    // Compare
    fprintf(stderr, "\nComparing results:\n");
    bool match = compare_tensors(result.data(), expected.data(), result.size(), 1e-4f, 1e-5f);

    ggml_free(ctx0);
    magpie_free(ctx);

    if (match) {
        fprintf(stderr, "\nSUCCESS: Text embedding matches reference!\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAILURE: Text embedding does not match reference\n");
        return 1;
    }
}
