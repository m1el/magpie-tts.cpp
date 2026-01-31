// Debug test for local transformer embedding/projection
#include "../src/magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = mctx->model.hparams;
    auto & lt = mctx->model.local_transformer;
    auto & emb = mctx->model.embeddings;

    // Test: Embed code 293 from codebook 0 and project to lt_dim
    int32_t code = 293;

    size_t ctx_size = ggml_tensor_overhead() * 32 + 8 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Input: single code index
    struct ggml_tensor * code_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code_idx, "code_idx");
    ggml_set_input(code_idx);

    // Lookup embedding from audio_embeddings[0]
    struct ggml_tensor * code_emb = ggml_get_rows(ctx0, emb.audio_emb_w[0], code_idx);
    ggml_set_name(code_emb, "code_emb");
    // code_emb is [d_model, 1], we want [d_model]
    struct ggml_tensor * code_emb_1d = ggml_reshape_1d(ctx0, code_emb, hp.d_model);

    // Project to lt_dim
    struct ggml_tensor * code_proj = ggml_mul_mat(ctx0, lt.in_proj_w, code_emb_1d);
    code_proj = ggml_add(ctx0, code_proj, lt.in_proj_b);
    ggml_set_name(code_proj, "code_proj");
    ggml_set_output(code_proj);

    // Also output raw embedding
    ggml_set_output(code_emb);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, code_proj);
    ggml_build_forward_expand(gf, code_emb);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(code_idx, &code, 0, sizeof(int32_t));
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get results
    std::vector<float> emb_result(hp.d_model);
    std::vector<float> proj_result(hp.lt_dim);
    ggml_backend_tensor_get(code_emb, emb_result.data(), 0, hp.d_model * sizeof(float));
    ggml_backend_tensor_get(code_proj, proj_result.data(), 0, hp.lt_dim * sizeof(float));

    printf("\n=== Code 293 Embedding (codebook 0) ===\n");
    printf("GGML first5:     %.6f %.6f %.6f %.6f %.6f\n",
           emb_result[0], emb_result[1], emb_result[2], emb_result[3], emb_result[4]);
    printf("PyTorch first5:  0.121225 0.490678 -0.284301 0.452872 -1.228324\n");

    printf("\n=== Projected Embedding ===\n");
    printf("GGML first5:     %.6f %.6f %.6f %.6f %.6f\n",
           proj_result[0], proj_result[1], proj_result[2], proj_result[3], proj_result[4]);
    printf("PyTorch first5:  0.316868 0.344950 -0.502660 -0.400141 0.118547\n");

    // Check match
    float expected_emb[] = {0.12122514f, 0.49067816f, -0.28430125f, 0.4528722f, -1.2283235f};
    float expected_proj[] = {0.31686807f, 0.34494963f, -0.50266045f, -0.40014094f, 0.11854737f};

    float max_emb_diff = 0;
    for (int i = 0; i < 5; i++) {
        float diff = fabsf(emb_result[i] - expected_emb[i]);
        if (diff > max_emb_diff) max_emb_diff = diff;
    }

    float max_proj_diff = 0;
    for (int i = 0; i < 5; i++) {
        float diff = fabsf(proj_result[i] - expected_proj[i]);
        if (diff > max_proj_diff) max_proj_diff = diff;
    }

    printf("\n=== Diff ===\n");
    printf("Embedding max diff (first5): %.6f\n", max_emb_diff);
    printf("Projected max diff (first5): %.6f\n", max_proj_diff);

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_emb_diff > 0.01f || max_proj_diff > 0.01f) {
        printf("\nFAILED: Large difference in embedding/projection\n");
        return 1;
    }

    printf("\nPASSED: Embedding and projection match\n");
    return 0;
}
