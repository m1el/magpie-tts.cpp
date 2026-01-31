// Debug decoder weight loading
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
    auto & dec_layer0 = mctx->model.decoder.layers[0];
    auto & enc_layer0 = mctx->model.encoder.layers[0];

    printf("\n=== Model Hyperparameters ===\n");
    printf("d_model=%d, dec_sa_heads=%d, dec_xa_heads=%d, dec_xa_d_head=%d\n",
           hp.d_model, hp.dec_sa_heads, hp.dec_xa_heads, hp.dec_xa_d_head);

    // Check decoder layer 0 weights
    printf("\n=== Decoder Layer 0 Weights ===\n");

    auto print_tensor_info = [&](const char* name, ggml_tensor* t) {
        if (!t) {
            printf("%s: NULL\n", name);
            return;
        }
        printf("%s: [%lld, %lld, %lld, %lld], type=%s\n",
               name, (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3],
               ggml_type_name(t->type));

        // Print first few values
        std::vector<float> data(std::min((int64_t)10, t->ne[0] * t->ne[1]));
        ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
        printf("  first5: %.6f %.6f %.6f %.6f %.6f\n",
               data[0], data[1], data[2], data[3], data[4]);
    };

    print_tensor_info("norm_self_w", dec_layer0.norm_self_w);
    print_tensor_info("sa_qkv_w", dec_layer0.sa_qkv_w);
    print_tensor_info("sa_out_w", dec_layer0.sa_out_w);
    print_tensor_info("norm_xa_q_w", dec_layer0.norm_xa_q_w);
    print_tensor_info("xa_q_w", dec_layer0.xa_q_w);
    print_tensor_info("xa_kv_w", dec_layer0.xa_kv_w);
    print_tensor_info("xa_out_w", dec_layer0.xa_out_w);
    print_tensor_info("norm_xa_mem_w", dec_layer0.norm_xa_mem_w);
    print_tensor_info("norm_ff_w", dec_layer0.norm_ff_w);
    print_tensor_info("ff_proj_w", dec_layer0.ff_proj_w);
    print_tensor_info("ff_out_w", dec_layer0.ff_out_w);

    printf("\n=== Encoder Layer 0 Weights (for comparison) ===\n");
    print_tensor_info("enc norm_self_w", enc_layer0.norm_self_w);
    print_tensor_info("enc sa_qkv_w", enc_layer0.sa_qkv_w);
    print_tensor_info("enc ff_proj_w", enc_layer0.ff_proj_w);
    print_tensor_info("enc ff_out_w", enc_layer0.ff_out_w);

    magpie_free(mctx);
    return 0;
}
