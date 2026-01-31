// Debug test for local transformer sequence construction
#include "../src/magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return {}; }
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

    // Load reference data
    int64_t proj_shape[4], hidden2_shape[4], logits1_shape[4];
    auto proj_data = load_reference("test_data/reference/lt_input_projected.bin", proj_shape);
    auto hidden2_expected = load_reference("test_data/reference/lt_hidden_seq2.bin", hidden2_shape);
    auto logits1_expected = load_reference("test_data/reference/lt_full_logits_cb1.bin", logits1_shape);

    if (proj_data.empty() || hidden2_expected.empty() || logits1_expected.empty()) {
        fprintf(stderr, "Failed to load reference data\n");
        magpie_free(mctx);
        return 1;
    }

    printf("lt_dim=%d, hidden2 shape=[%lld,%lld]\n", hp.lt_dim,
           (long long)hidden2_shape[0], (long long)hidden2_shape[1]);

    // Expected values from PyTorch for sequence position 1 (code 293 projected)
    float expected_pos1_before_pos_emb[] = {0.31686807f, 0.34494963f, -0.50266045f, -0.40014094f, 0.11854737f};
    float expected_pos1_after_pos_emb[] = {0.22848994f, 0.6719235f, -0.70018435f, -0.15899327f, 0.09095154f};

    // Step 1: Build the 2-position sequence
    // Position 0: projected decoder hidden (from reference)
    // Position 1: projected code 293 embedding

    std::vector<float> lt_seq_data(hp.lt_dim * 2);

    // Copy position 0 (projected decoder hidden)
    for (int i = 0; i < hp.lt_dim; i++) {
        lt_seq_data[i] = proj_data[i];
    }

    // Compute position 1 (embed code 293 and project)
    {
        size_t ctx_size = ggml_tensor_overhead() * 16 + 4 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        int32_t code = 293;
        struct ggml_tensor * code_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(code_idx, "code_idx");
        ggml_set_input(code_idx);

        struct ggml_tensor * code_emb = ggml_get_rows(ctx0, emb.audio_emb_w[0], code_idx);
        struct ggml_tensor * code_emb_1d = ggml_reshape_1d(ctx0, code_emb, hp.d_model);
        struct ggml_tensor * code_proj = ggml_mul_mat(ctx0, lt.in_proj_w, code_emb_1d);
        code_proj = ggml_add(ctx0, code_proj, lt.in_proj_b);
        ggml_set_name(code_proj, "output");
        ggml_set_output(code_proj);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, code_proj);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(code_idx, &code, 0, sizeof(int32_t));
        ggml_backend_graph_compute(mctx->model.backend, gf);

        std::vector<float> proj_result(hp.lt_dim);
        ggml_backend_tensor_get(code_proj, proj_result.data(), 0, hp.lt_dim * sizeof(float));

        // Copy to position 1
        for (int i = 0; i < hp.lt_dim; i++) {
            lt_seq_data[hp.lt_dim + i] = proj_result[i];
        }

        printf("\n=== Position 1 (before pos emb) ===\n");
        printf("GGML first5:     %.6f %.6f %.6f %.6f %.6f\n",
               proj_result[0], proj_result[1], proj_result[2], proj_result[3], proj_result[4]);
        printf("PyTorch first5:  %.6f %.6f %.6f %.6f %.6f\n",
               expected_pos1_before_pos_emb[0], expected_pos1_before_pos_emb[1],
               expected_pos1_before_pos_emb[2], expected_pos1_before_pos_emb[3],
               expected_pos1_before_pos_emb[4]);

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }

    // Step 2: Add position embeddings and run through transformer
    {
        size_t ctx_size = ggml_tensor_overhead() * 256 + 64 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * seq_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.lt_dim, 2);
        ggml_set_name(seq_input, "seq_input");
        ggml_set_input(seq_input);

        // Add position embeddings for positions 0 and 1
        struct ggml_tensor * pos_slice = ggml_view_2d(ctx0, lt.pos_emb_w,
            hp.lt_dim, 2, lt.pos_emb_w->nb[1], 0);
        struct ggml_tensor * with_pos = ggml_add(ctx0, seq_input, pos_slice);

        ggml_set_name(with_pos, "with_pos");
        ggml_set_output(with_pos);

        // Build graph
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, with_pos);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(seq_input, lt_seq_data.data(), 0, hp.lt_dim * 2 * sizeof(float));
        ggml_backend_graph_compute(mctx->model.backend, gf);

        std::vector<float> with_pos_data(hp.lt_dim * 2);
        ggml_backend_tensor_get(with_pos, with_pos_data.data(), 0, hp.lt_dim * 2 * sizeof(float));

        printf("\n=== Position 1 (after pos emb) ===\n");
        printf("GGML first5:     %.6f %.6f %.6f %.6f %.6f\n",
               with_pos_data[hp.lt_dim+0], with_pos_data[hp.lt_dim+1],
               with_pos_data[hp.lt_dim+2], with_pos_data[hp.lt_dim+3],
               with_pos_data[hp.lt_dim+4]);
        printf("PyTorch first5:  %.6f %.6f %.6f %.6f %.6f\n",
               expected_pos1_after_pos_emb[0], expected_pos1_after_pos_emb[1],
               expected_pos1_after_pos_emb[2], expected_pos1_after_pos_emb[3],
               expected_pos1_after_pos_emb[4]);

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }

    // Step 3: Run full transformer layer and compare
    {
        size_t ctx_size = ggml_tensor_overhead() * 256 + 64 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * seq_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.lt_dim, 2);
        ggml_set_name(seq_input, "seq_input");
        ggml_set_input(seq_input);

        // Add position embeddings
        struct ggml_tensor * pos_slice = ggml_view_2d(ctx0, lt.pos_emb_w,
            hp.lt_dim, 2, lt.pos_emb_w->nb[1], 0);
        struct ggml_tensor * with_pos = ggml_add(ctx0, seq_input, pos_slice);

        // Norm -> Self-attention -> Residual
        struct ggml_tensor * residual = with_pos;
        struct ggml_tensor * x = magpie_build_layer_norm(ctx0, with_pos, lt.norm_self_w, hp.eps);

        // Create causal mask
        struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 2);
        ggml_set_name(mask, "causal_mask");
        ggml_set_input(mask);

        x = magpie_build_self_attention_with_mask(ctx0, x, lt.sa_qkv_w, lt.sa_out_w,
                                                   hp.lt_heads, true, mask);
        x = ggml_add(ctx0, x, residual);

        // Norm -> FFN -> Residual
        residual = x;
        x = magpie_build_layer_norm(ctx0, x, lt.norm_ff_w, hp.eps);
        x = magpie_build_conv_ffn(ctx0, x, lt.ff_proj_w, lt.ff_out_w, 1);
        x = ggml_add(ctx0, x, residual);

        ggml_set_name(x, "layer_out");
        ggml_set_output(x);

        // Output projection for codebook 1
        struct ggml_tensor * last_hidden = ggml_view_1d(ctx0, x, hp.lt_dim, hp.lt_dim * sizeof(float));
        last_hidden = ggml_cont(ctx0, last_hidden);
        struct ggml_tensor * logits = ggml_mul_mat(ctx0, lt.out_proj_w[1], last_hidden);
        logits = ggml_add(ctx0, logits, lt.out_proj_b[1]);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
        ggml_build_forward_expand(gf, logits);
        ggml_build_forward_expand(gf, x);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(seq_input, lt_seq_data.data(), 0, hp.lt_dim * 2 * sizeof(float));

        // Set causal mask
        float mask_data[4];
        fill_causal_mask_f32(mask_data, 2);
        ggml_backend_tensor_set(mask, mask_data, 0, 4 * sizeof(float));

        ggml_backend_graph_compute(mctx->model.backend, gf);

        std::vector<float> layer_out(hp.lt_dim * 2);
        ggml_backend_tensor_get(x, layer_out.data(), 0, hp.lt_dim * 2 * sizeof(float));

        printf("\n=== After transformer layer ===\n");
        printf("GGML pos0 first5:     %.6f %.6f %.6f %.6f %.6f\n",
               layer_out[0], layer_out[1], layer_out[2], layer_out[3], layer_out[4]);
        printf("PyTorch pos0 first5:  %.6f %.6f %.6f %.6f %.6f\n",
               hidden2_expected[0], hidden2_expected[1], hidden2_expected[2],
               hidden2_expected[3], hidden2_expected[4]);
        printf("GGML pos1 first5:     %.6f %.6f %.6f %.6f %.6f\n",
               layer_out[hp.lt_dim+0], layer_out[hp.lt_dim+1],
               layer_out[hp.lt_dim+2], layer_out[hp.lt_dim+3], layer_out[hp.lt_dim+4]);
        printf("PyTorch pos1 first5:  %.6f %.6f %.6f %.6f %.6f\n",
               hidden2_expected[hp.lt_dim+0], hidden2_expected[hp.lt_dim+1],
               hidden2_expected[hp.lt_dim+2], hidden2_expected[hp.lt_dim+3],
               hidden2_expected[hp.lt_dim+4]);

        std::vector<float> logits_data(hp.vocab_per_cb);
        ggml_backend_tensor_get(logits, logits_data.data(), 0, hp.vocab_per_cb * sizeof(float));

        int ggml_argmax = 0, pytorch_argmax = 0;
        float ggml_max = logits_data[0], pytorch_max = logits1_expected[0];
        for (int i = 1; i < hp.vocab_per_cb; i++) {
            if (logits_data[i] > ggml_max) { ggml_max = logits_data[i]; ggml_argmax = i; }
            if (logits1_expected[i] > pytorch_max) { pytorch_max = logits1_expected[i]; pytorch_argmax = i; }
        }

        printf("\n=== Codebook 1 Logits ===\n");
        printf("GGML argmax:    %d (value: %.6f)\n", ggml_argmax, ggml_max);
        printf("PyTorch argmax: %d (value: %.6f)\n", pytorch_argmax, pytorch_max);

        float max_diff = 0;
        for (int i = 0; i < hp.vocab_per_cb; i++) {
            float diff = fabsf(logits_data[i] - logits1_expected[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max logits diff: %.6f\n", max_diff);

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);

        if (ggml_argmax == pytorch_argmax) {
            printf("\nPASSED: Codebook 1 argmax matches!\n");
        } else {
            printf("\nFAILED: Codebook 1 argmax mismatch\n");
            magpie_free(mctx);
            return 1;
        }
    }

    magpie_free(mctx);
    return 0;
}
