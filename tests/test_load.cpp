#include "magpie.h"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    fprintf(stderr, "Loading model from: %s\n", model_path);

    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = ctx->model.hparams;
    fprintf(stderr, "\nModel hyperparameters:\n");
    fprintf(stderr, "  d_model:         %d\n", hp.d_model);
    fprintf(stderr, "  d_ffn:           %d\n", hp.d_ffn);
    fprintf(stderr, "  d_head:          %d\n", hp.d_head);
    fprintf(stderr, "  enc_layers:      %d\n", hp.enc_layers);
    fprintf(stderr, "  enc_heads:       %d\n", hp.enc_heads);
    fprintf(stderr, "  dec_layers:      %d\n", hp.dec_layers);
    fprintf(stderr, "  dec_sa_heads:    %d\n", hp.dec_sa_heads);
    fprintf(stderr, "  dec_xa_heads:    %d\n", hp.dec_xa_heads);
    fprintf(stderr, "  lt_dim:          %d\n", hp.lt_dim);
    fprintf(stderr, "  text_vocab_size: %d\n", hp.text_vocab_size);
    fprintf(stderr, "  num_codebooks:   %d\n", hp.num_codebooks);
    fprintf(stderr, "  vocab_per_cb:    %d\n", hp.vocab_per_cb);

    fprintf(stderr, "\nLoaded tensors:\n");
    int count = 0;
    for (const auto & [name, tensor] : ctx->model.tensors) {
        fprintf(stderr, "  %s: [", name.c_str());
        for (int i = 0; i < ggml_n_dims(tensor); i++) {
            if (i > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%lld", (long long)tensor->ne[i]);
        }
        fprintf(stderr, "] (%s)\n", ggml_type_name(tensor->type));
        count++;
        if (count >= 20) {
            fprintf(stderr, "  ... and %d more\n", (int)ctx->model.tensors.size() - 20);
            break;
        }
    }

    fprintf(stderr, "\nKV cache:\n");
    fprintf(stderr, "  max_seq:     %d\n", ctx->state.kv_cache.max_seq);
    fprintf(stderr, "  k_cache:     %zu layers\n", ctx->state.kv_cache.k_cache.size());
    fprintf(stderr, "  xa_k_cache:  %zu layers\n", ctx->state.kv_cache.xa_k_cache.size());

    // Validate key tensors are mapped
    fprintf(stderr, "\nValidating tensor mappings:\n");
    int errors = 0;

    #define CHECK_TENSOR(ptr, name) \
        if (!(ptr)) { fprintf(stderr, "  MISSING: %s\n", name); errors++; } \
        else { fprintf(stderr, "  OK: %s\n", name); }

    CHECK_TENSOR(ctx->model.embeddings.text_emb_w, "text_embedding");
    for (int i = 0; i < 8; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "audio_embeddings.%d", i);
        CHECK_TENSOR(ctx->model.embeddings.audio_emb_w[i], buf);
    }
    CHECK_TENSOR(ctx->model.embeddings.baked_context_w, "baked_context_embedding");

    CHECK_TENSOR(ctx->model.encoder.pos_emb_w, "encoder.pos_emb");
    CHECK_TENSOR(ctx->model.encoder.norm_out_w, "encoder.norm");
    for (int i = 0; i < hp.enc_layers; i++) {
        const auto & layer = ctx->model.encoder.layers[i];
        char buf[64];
        snprintf(buf, sizeof(buf), "encoder.layer.%d.norm_self", i);
        CHECK_TENSOR(layer.norm_self_w, buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.sa_qkv", i);
        CHECK_TENSOR(layer.sa_qkv_w, buf);
    }

    CHECK_TENSOR(ctx->model.decoder.pos_emb_w, "decoder.pos_emb");
    CHECK_TENSOR(ctx->model.decoder.norm_out_w, "decoder.norm");
    for (int i = 0; i < hp.dec_layers; i++) {
        const auto & layer = ctx->model.decoder.layers[i];
        char buf[64];
        snprintf(buf, sizeof(buf), "decoder.layer.%d.norm_self", i);
        CHECK_TENSOR(layer.norm_self_w, buf);
        snprintf(buf, sizeof(buf), "decoder.layer.%d.xa_q", i);
        CHECK_TENSOR(layer.xa_q_w, buf);
    }

    CHECK_TENSOR(ctx->model.final_proj.weight, "final_proj.weight");
    CHECK_TENSOR(ctx->model.final_proj.bias, "final_proj.bias");

    CHECK_TENSOR(ctx->model.local_transformer.in_proj_w, "local_transformer.in_proj");
    CHECK_TENSOR(ctx->model.local_transformer.pos_emb_w, "local_transformer.pos_emb");
    CHECK_TENSOR(ctx->model.local_transformer.norm_self_w, "local_transformer.norm_self");
    for (int i = 0; i < 8; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "local_transformer.out_proj.%d.w", i);
        CHECK_TENSOR(ctx->model.local_transformer.out_proj_w[i], buf);
    }

    #undef CHECK_TENSOR

    if (errors > 0) {
        fprintf(stderr, "\n%d tensor mapping errors!\n", errors);
    }

    magpie_free(ctx);
    fprintf(stderr, "\nSuccess!\n");
    return errors > 0 ? 1 : 0;
}
