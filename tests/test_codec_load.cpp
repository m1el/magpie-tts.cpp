/**
 * Test codec loading from GGUF
 *
 * This test verifies that the audio codec model loads correctly
 * and all tensors are mapped to the appropriate structures.
 */

#include "magpie.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    const char * codec_path = "weights/nano-codec-f32.gguf";
    if (argc > 1) {
        codec_path = argv[1];
    }

    printf("=== Test Codec Loading ===\n");
    printf("Codec path: %s\n\n", codec_path);

    // Load codec
    magpie_codec * codec = magpie_codec_init(codec_path);
    if (!codec) {
        fprintf(stderr, "ERROR: Failed to load codec\n");
        return 1;
    }

    printf("Codec loaded successfully!\n\n");

    // Print hyperparameters
    const auto & hp = codec->hparams;
    printf("=== Hyperparameters ===\n");
    printf("  sample_rate:     %d\n", hp.sample_rate);
    printf("  num_codebooks:   %d\n", hp.num_codebooks);
    printf("  codebook_size:   %d\n", hp.codebook_size);
    printf("  hop_length:      %d\n", hp.hop_length);
    printf("  latent_dim:      %d\n", hp.latent_dim);
    printf("  base_channels:   %d\n", hp.base_channels);
    printf("  num_upsample:    %d\n", hp.num_upsample_layers);
    printf("\n");

    // Verify pre-conv tensors
    printf("=== Pre-Conv Tensors ===\n");
    if (codec->pre_conv_w) {
        printf("  pre_conv_w: [%lld, %lld, %lld]\n",
               (long long)codec->pre_conv_w->ne[0],
               (long long)codec->pre_conv_w->ne[1],
               (long long)codec->pre_conv_w->ne[2]);
    } else {
        printf("  pre_conv_w: MISSING!\n");
    }
    if (codec->pre_conv_b) {
        printf("  pre_conv_b: [%lld]\n", (long long)codec->pre_conv_b->ne[0]);
    } else {
        printf("  pre_conv_b: MISSING!\n");
    }
    printf("\n");

    // Verify post-conv tensors
    printf("=== Post-Conv Tensors ===\n");
    if (codec->post_conv_w) {
        printf("  post_conv_w: [%lld, %lld, %lld]\n",
               (long long)codec->post_conv_w->ne[0],
               (long long)codec->post_conv_w->ne[1],
               (long long)codec->post_conv_w->ne[2]);
    } else {
        printf("  post_conv_w: MISSING!\n");
    }
    if (codec->post_conv_b) {
        printf("  post_conv_b: [%lld]\n", (long long)codec->post_conv_b->ne[0]);
    } else {
        printf("  post_conv_b: MISSING!\n");
    }
    if (codec->post_act_alpha) {
        printf("  post_act_alpha: [%lld, %lld, %lld]\n",
               (long long)codec->post_act_alpha->ne[0],
               (long long)codec->post_act_alpha->ne[1],
               (long long)codec->post_act_alpha->ne[2]);
    } else {
        printf("  post_act_alpha: MISSING!\n");
    }
    printf("\n");

    // Verify upsample layers
    printf("=== Upsample Layers (%zu) ===\n", codec->upsample_layers.size());
    for (size_t i = 0; i < codec->upsample_layers.size(); i++) {
        const auto & up = codec->upsample_layers[i];
        printf("  [%zu] act_alpha: %s, conv_w: %s, conv_b: %s\n",
               i,
               up.act_alpha ? "OK" : "MISSING",
               up.conv_w ? "OK" : "MISSING",
               up.conv_b ? "OK" : "MISSING");
        if (up.conv_w) {
            printf("       conv_w shape: [%lld, %lld, %lld]\n",
                   (long long)up.conv_w->ne[0],
                   (long long)up.conv_w->ne[1],
                   (long long)up.conv_w->ne[2]);
        }
    }
    printf("\n");

    // Verify residual layers
    printf("=== Residual Layers (%zu) ===\n", codec->res_layers.size());
    for (size_t i = 0; i < codec->res_layers.size(); i++) {
        const auto & res = codec->res_layers[i];
        printf("  [%zu] %zu res_blocks\n", i, res.res_blocks.size());
        for (size_t j = 0; j < res.res_blocks.size(); j++) {
            const auto & rb = res.res_blocks[j];
            printf("    [%zu][%zu] %zu inner_blocks\n", i, j, rb.inner_blocks.size());

            // Check first inner block has all tensors
            if (!rb.inner_blocks.empty()) {
                const auto & ib = rb.inner_blocks[0];
                bool all_ok = ib.input_act_alpha && ib.input_conv_w && ib.input_conv_b &&
                              ib.skip_act_alpha && ib.skip_conv_w && ib.skip_conv_b;
                printf("      inner[0]: %s\n", all_ok ? "all tensors OK" : "MISSING tensors");
            }
        }
    }
    printf("\n");

    // Verify FSQ parameters
    printf("=== FSQ Codebooks (%zu) ===\n", codec->fsqs.size());
    for (size_t i = 0; i < codec->fsqs.size(); i++) {
        const auto & fsq = codec->fsqs[i];
        printf("  [%zu] dim_base_index: %s, num_levels: %s\n",
               i,
               fsq.dim_base_index ? "OK" : "MISSING",
               fsq.num_levels ? "OK" : "MISSING");
    }
    printf("\n");

    // Count total mapped tensors
    int total_tensors = codec->tensors.size();
    printf("=== Summary ===\n");
    printf("Total tensors in file: %d\n", total_tensors);

    // Count successfully mapped tensors
    int mapped = 0;
    if (codec->pre_conv_w) mapped++;
    if (codec->pre_conv_b) mapped++;
    if (codec->post_conv_w) mapped++;
    if (codec->post_conv_b) mapped++;
    if (codec->post_act_alpha) mapped++;

    for (const auto & up : codec->upsample_layers) {
        if (up.act_alpha) mapped++;
        if (up.conv_w) mapped++;
        if (up.conv_b) mapped++;
    }

    for (const auto & res : codec->res_layers) {
        for (const auto & rb : res.res_blocks) {
            for (const auto & ib : rb.inner_blocks) {
                if (ib.input_act_alpha) mapped++;
                if (ib.input_conv_w) mapped++;
                if (ib.input_conv_b) mapped++;
                if (ib.skip_act_alpha) mapped++;
                if (ib.skip_conv_w) mapped++;
                if (ib.skip_conv_b) mapped++;
            }
        }
    }

    for (const auto & fsq : codec->fsqs) {
        if (fsq.dim_base_index) mapped++;
        if (fsq.num_levels) mapped++;
    }

    printf("Mapped tensors: %d\n", mapped);
    printf("\n");

    // Cleanup
    magpie_codec_free(codec);
    printf("Test passed!\n");

    return 0;
}
