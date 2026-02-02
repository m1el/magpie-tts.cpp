#include "magpie.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>

//
// Backend initialization helpers
//

static bool init_backend(magpie_model & model, magpie_backend_type backend) {
    model.backend_type = backend;

    if (backend == MAGPIE_BACKEND_AUTO) {
#ifdef GGML_USE_CUDA
        backend = MAGPIE_BACKEND_CUDA;
#elif defined(GGML_USE_METAL)
        backend = MAGPIE_BACKEND_METAL;
#else
        backend = MAGPIE_BACKEND_CPU;
#endif
        model.backend_type = backend;
    }

    switch (backend) {
        case MAGPIE_BACKEND_CUDA:
#ifdef GGML_USE_CUDA
            model.backend = ggml_backend_cuda_init(0);
            if (!model.backend) {
                fprintf(stderr, "magpie: failed to init CUDA backend, falling back to CPU\n");
                model.backend = ggml_backend_cpu_init();
                model.backend_type = MAGPIE_BACKEND_CPU;
            }
#else
            fprintf(stderr, "magpie: CUDA not compiled in, using CPU\n");
            model.backend = ggml_backend_cpu_init();
            model.backend_type = MAGPIE_BACKEND_CPU;
#endif
            break;

        case MAGPIE_BACKEND_METAL:
#ifdef GGML_USE_METAL
            model.backend = ggml_backend_metal_init();
            if (!model.backend) {
                fprintf(stderr, "magpie: failed to init Metal backend, falling back to CPU\n");
                model.backend = ggml_backend_cpu_init();
                model.backend_type = MAGPIE_BACKEND_CPU;
            }
#else
            fprintf(stderr, "magpie: Metal not compiled in, using CPU\n");
            model.backend = ggml_backend_cpu_init();
            model.backend_type = MAGPIE_BACKEND_CPU;
#endif
            break;

        case MAGPIE_BACKEND_CPU:
        default:
            model.backend = ggml_backend_cpu_init();
            model.backend_type = MAGPIE_BACKEND_CPU;
            break;
    }

    return model.backend != nullptr;
}

//
// Hyperparameter loading from GGUF
//

static void read_hparams(gguf_context * gguf_ctx, magpie_hparams & hparams) {
    auto get_i32 = [&](const char * key, int32_t def) -> int32_t {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? (int32_t)gguf_get_val_u32(gguf_ctx, idx) : def;
    };

    auto get_f32 = [&](const char * key, float def) -> float {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? gguf_get_val_f32(gguf_ctx, idx) : def;
    };

    // Read hyperparameters with defaults from struct
    hparams.d_model         = get_i32("magpie.d_model", hparams.d_model);
    hparams.d_ffn           = get_i32("magpie.d_ffn", hparams.d_ffn);
    hparams.d_head          = get_i32("magpie.d_head", hparams.d_head);

    hparams.enc_layers      = get_i32("magpie.enc_layers", hparams.enc_layers);
    hparams.enc_heads       = get_i32("magpie.enc_heads", hparams.enc_heads);
    hparams.enc_kernel      = get_i32("magpie.enc_kernel", hparams.enc_kernel);

    hparams.dec_layers      = get_i32("magpie.dec_layers", hparams.dec_layers);
    hparams.dec_sa_heads    = get_i32("magpie.dec_sa_heads", hparams.dec_sa_heads);
    hparams.dec_xa_heads    = get_i32("magpie.dec_xa_heads", hparams.dec_xa_heads);
    hparams.dec_xa_d_head   = get_i32("magpie.dec_xa_d_head", hparams.dec_xa_d_head);
    hparams.dec_kernel      = get_i32("magpie.dec_kernel", hparams.dec_kernel);

    hparams.lt_dim          = get_i32("magpie.lt_dim", hparams.lt_dim);
    hparams.lt_ffn_dim      = get_i32("magpie.lt_ffn_dim", hparams.lt_ffn_dim);
    hparams.lt_layers       = get_i32("magpie.lt_layers", hparams.lt_layers);
    hparams.lt_heads        = get_i32("magpie.lt_heads", hparams.lt_heads);

    hparams.text_vocab_size = get_i32("magpie.text_vocab_size", hparams.text_vocab_size);
    hparams.num_codebooks   = get_i32("magpie.num_codebooks", hparams.num_codebooks);
    hparams.codebook_size   = get_i32("magpie.codebook_size", hparams.codebook_size);
    hparams.vocab_per_cb    = get_i32("magpie.vocab_per_cb", hparams.vocab_per_cb);

    hparams.num_speakers    = get_i32("magpie.num_speakers", hparams.num_speakers);
    hparams.context_frames  = get_i32("magpie.context_frames", hparams.context_frames);

    hparams.text_bos_id     = get_i32("magpie.text_bos_id", hparams.text_bos_id);
    hparams.text_eos_id     = get_i32("magpie.text_eos_id", hparams.text_eos_id);
    hparams.audio_bos_id    = get_i32("magpie.audio_bos_id", hparams.audio_bos_id);
    hparams.audio_eos_id    = get_i32("magpie.audio_eos_id", hparams.audio_eos_id);

    hparams.max_dec_steps   = get_i32("magpie.max_dec_steps", hparams.max_dec_steps);
    hparams.sample_rate     = get_i32("magpie.sample_rate", hparams.sample_rate);

    hparams.eps             = get_f32("magpie.eps", hparams.eps);
}

//
// Tokenizer
//

// Helper to split string by delimiter
static std::vector<std::string> split_string(const std::string & str, char delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delim);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delim, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

// Convert string to lowercase
static std::string to_lower(const std::string & str) {
    std::string result = str;
    for (char & c : result) {
        if (c >= 'A' && c <= 'Z') {
            c = c - 'A' + 'a';
        }
    }
    return result;
}

// Convert number to words (matches NeMo text normalization)
static std::string number_to_words(int64_t n, bool use_and = true) {
    if (n < 0) return "minus " + number_to_words(-n, use_and);

    static const char * ones[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"
    };
    static const char * tens[] = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    };

    if (n < 20) return ones[n];
    if (n < 100) {
        std::string s = tens[n / 10];
        if (n % 10 != 0) s += " " + std::string(ones[n % 10]);
        return s;
    }
    if (n < 1000) {
        std::string s = std::string(ones[n / 100]) + " hundred";
        if (n % 100 != 0) {
            if (use_and) s += " and";
            s += " " + number_to_words(n % 100, use_and);
        }
        return s;
    }
    if (n < 10000) {
        std::string s = number_to_words(n / 1000, use_and) + " thousand";
        if (n % 1000 != 0) s += " " + number_to_words(n % 1000, use_and);
        return s;
    }
    // For larger numbers, NeMo often reads digit pairs (year-style)
    // 10000+ -> read digit by digit
    if (n < 1000000) {
        // Try to read as digit pairs if it's a round-ish number
        // Otherwise fall back to full expansion
        std::string s = number_to_words(n / 1000, use_and) + " thousand";
        if (n % 1000 != 0) s += " " + number_to_words(n % 1000, use_and);
        return s;
    }
    if (n < 1000000000) {
        std::string s = number_to_words(n / 1000000, use_and) + " million";
        if (n % 1000000 != 0) s += " " + number_to_words(n % 1000000, use_and);
        return s;
    }
    if (n < 1000000000000LL) {
        std::string s = number_to_words(n / 1000000000, use_and) + " billion";
        if (n % 1000000000 != 0) s += " " + number_to_words(n % 1000000000, use_and);
        return s;
    }
    // Fallback for very large numbers
    return std::to_string(n);
}

// Convert 4-digit year to words (2024 -> "twenty twenty four")
static std::string year_to_words(int64_t n) {
    if (n < 1000 || n > 9999) return number_to_words(n);

    int high = n / 100;
    int low = n % 100;

    if (low == 0) {
        // 1900 -> "nineteen hundred"
        return number_to_words(high) + " hundred";
    } else if (low < 10) {
        // 2001 -> "two thousand one" (not year format)
        return number_to_words(n);
    } else {
        // 2024 -> "twenty twenty four"
        return number_to_words(high) + " " + number_to_words(low);
    }
}

// Convert ordinal number to words (1st -> first, 2nd -> second, etc.)
static std::string ordinal_to_words(int64_t n) {
    // Special cases for 1-12
    static const char * special[] = {
        "", "first", "second", "third", "fourth", "fifth", "sixth",
        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"
    };
    if (n >= 1 && n <= 12) return special[n];

    // For others, convert to cardinal and add suffix
    std::string cardinal = number_to_words(n);

    // Handle teens (thirteenth, fourteenth, etc.)
    if (n >= 13 && n <= 19) return cardinal + "th";

    // Handle tens (twentieth, thirtieth, etc.)
    if (n % 10 == 0 && n >= 20 && n < 100) {
        // Remove 'y' and add 'ieth'
        if (cardinal.size() > 0 && cardinal.back() == 'y') {
            return cardinal.substr(0, cardinal.size() - 1) + "ieth";
        }
        return cardinal + "th";
    }

    // For compound numbers like 21st, 32nd, etc.
    int ones_digit = n % 10;
    if (ones_digit == 1) return cardinal.substr(0, cardinal.rfind(' ') + 1) + "first";
    if (ones_digit == 2) return cardinal.substr(0, cardinal.rfind(' ') + 1) + "second";
    if (ones_digit == 3) return cardinal.substr(0, cardinal.rfind(' ') + 1) + "third";

    return cardinal + "th";
}

// Normalize text: expand numbers, handle ordinals, currency, percent (matches NeMo behavior)
static std::string normalize_text(const std::string & text) {
    std::string result;
    result.reserve(text.size() * 2);

    size_t i = 0;
    while (i < text.size()) {
        // Check for currency ($50 -> "fifty dollars")
        if (text[i] == '$' && i + 1 < text.size() && text[i+1] >= '0' && text[i+1] <= '9') {
            i++;  // Skip $
            int64_t num = 0;
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                num = num * 10 + (text[i] - '0');
                i++;
            }
            result += number_to_words(num) + " dollar";
            if (num != 1) result += "s";
            continue;
        }

        // Check for numbers (including negative)
        if ((text[i] >= '0' && text[i] <= '9') ||
            (text[i] == '-' && i + 1 < text.size() && text[i+1] >= '0' && text[i+1] <= '9')) {

            bool negative = false;
            if (text[i] == '-') {
                negative = true;
                i++;
            }

            // Parse the number
            int64_t num = 0;
            int num_digits = 0;
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                num = num * 10 + (text[i] - '0');
                num_digits++;
                i++;
            }

            // Check for percent (50% -> "fifty percent")
            if (i < text.size() && text[i] == '%') {
                i++;  // Skip %
                std::string num_words = number_to_words(num);
                if (negative) num_words = "minus " + num_words;
                result += num_words + " percent";
                continue;
            }

            // Check for ordinal suffix (st, nd, rd, th)
            bool is_ordinal = false;
            if (i + 1 < text.size()) {
                char c1 = text[i];
                char c2 = text[i + 1];
                // Convert to lowercase for comparison
                if (c1 >= 'A' && c1 <= 'Z') c1 = c1 - 'A' + 'a';
                if (c2 >= 'A' && c2 <= 'Z') c2 = c2 - 'A' + 'a';

                if ((c1 == 's' && c2 == 't') ||  // 1st, 21st, etc.
                    (c1 == 'n' && c2 == 'd') ||  // 2nd, 22nd, etc.
                    (c1 == 'r' && c2 == 'd') ||  // 3rd, 23rd, etc.
                    (c1 == 't' && c2 == 'h')) {  // 4th, 5th, etc.
                    is_ordinal = true;
                    i += 2;
                }
            }

            // Convert to words
            std::string num_words;
            if (is_ordinal) {
                num_words = ordinal_to_words(num);
            } else if (num_digits == 4 && num >= 1000 && num <= 2099) {
                // Likely a year - use year format
                num_words = year_to_words(num);
            } else {
                num_words = number_to_words(num);
            }
            if (negative && num != 0) {
                num_words = "minus " + num_words;
            }

            result += num_words;
            continue;
        }

        // Regular character
        result += text[i];
        i++;
    }

    return result;
}

bool magpie_tokenizer_init(magpie_tokenizer * tok, struct gguf_context * gguf_ctx) {
    // Load vocabulary (newline-separated string)
    int vocab_key = gguf_find_key(gguf_ctx, "magpie.tokenizer.vocab");
    if (vocab_key < 0) {
        fprintf(stderr, "magpie_tokenizer: vocabulary not found in model\n");
        return false;
    }

    const char * vocab_str = gguf_get_val_str(gguf_ctx, vocab_key);

    // Split by newlines
    tok->vocab = split_string(std::string(vocab_str), '\n');
    fprintf(stderr, "magpie_tokenizer: loaded %zu vocabulary tokens\n", tok->vocab.size());

    // Build reverse mapping
    for (size_t i = 0; i < tok->vocab.size(); i++) {
        tok->token_to_id[tok->vocab[i]] = (int32_t)i;
    }

    // Load dictionary (word\tpron\n format - standard TSV)
    int dict_key = gguf_find_key(gguf_ctx, "magpie.tokenizer.dict");
    if (dict_key >= 0) {
        const char * dict_str = gguf_get_val_str(gguf_ctx, dict_key);

        // Parse TSV: word\tpron per line
        std::vector<std::string> lines = split_string(std::string(dict_str), '\n');
        for (const std::string & line : lines) {
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                std::string word = line.substr(0, tab_pos);
                std::string pron = line.substr(tab_pos + 1);
                tok->dict[word] = pron;
            }
        }
        fprintf(stderr, "magpie_tokenizer: loaded %zu dictionary entries\n", tok->dict.size());
    }

    // Load special token IDs
    auto get_tok_id = [&](const char * key, int32_t def) -> int32_t {
        int idx = gguf_find_key(gguf_ctx, key);
        return (idx >= 0) ? gguf_get_val_u32(gguf_ctx, idx) : def;
    };

    tok->pad_id   = get_tok_id("magpie.tokenizer.pad", 94);
    tok->oov_id   = get_tok_id("magpie.tokenizer.oov", 95);
    tok->space_id = get_tok_id("magpie.tokenizer.space", 93);
    tok->bos_id   = get_tok_id("magpie.text_bos_id", 2378);
    tok->eos_id   = get_tok_id("magpie.text_eos_id", 2379);

    tok->loaded = true;
    return true;
}

std::vector<int32_t> magpie_tokenize(const magpie_tokenizer * tok, const std::string & text) {
    std::vector<int32_t> tokens;

    if (!tok || !tok->loaded) {
        fprintf(stderr, "magpie_tokenize: tokenizer not loaded\n");
        return tokens;
    }

    // Add BOS
    tokens.push_back(tok->bos_id);

    // Normalize text: expand numbers, currency, etc. then lowercase
    std::string normalized = to_lower(normalize_text(text));

    // Replace punctuation with space + punctuation + space for proper tokenization
    std::string processed;
    for (char c : normalized) {
        if (c == ',' || c == '.' || c == '!' || c == '?' || c == ':' || c == ';') {
            processed += ' ';
            processed += c;
            processed += ' ';
        } else {
            processed += c;
        }
    }

    // Split into words
    std::vector<std::string> words = split_string(processed, ' ');

    for (const std::string & word : words) {
        if (word.empty()) continue;

        // Check if it's punctuation (single char that's in vocab)
        if (word.size() == 1) {
            auto it = tok->token_to_id.find(word);
            if (it != tok->token_to_id.end()) {
                tokens.push_back(it->second);
                continue;
            }
        }

        // Look up word in pronunciation dictionary
        auto dict_it = tok->dict.find(word);
        if (dict_it != tok->dict.end()) {
            // Found pronunciation - tokenize the IPA string
            const std::string & pron = dict_it->second;
            for (size_t i = 0; i < pron.size(); ) {
                // Try to match longest token first (for multi-byte UTF-8 chars)
                bool found = false;
                for (size_t len = std::min(pron.size() - i, (size_t)4); len > 0; len--) {
                    std::string substr = pron.substr(i, len);
                    auto it = tok->token_to_id.find(substr);
                    if (it != tok->token_to_id.end()) {
                        tokens.push_back(it->second);
                        i += len;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // Skip unknown character
                    i++;
                }
            }
        } else {
            // OOV word - use character fallback (uppercase with # prefix if needed)
            for (char c : word) {
                char upper = (c >= 'a' && c <= 'z') ? (c - 'a' + 'A') : c;
                std::string char_tok(1, upper);
                auto it = tok->token_to_id.find(char_tok);
                if (it != tok->token_to_id.end()) {
                    tokens.push_back(it->second);
                }
            }
        }

        // Add space between words
        if (tok->space_id >= 0) {
            tokens.push_back(tok->space_id);
        }
    }

    // Remove trailing space if present
    if (!tokens.empty() && tokens.back() == tok->space_id) {
        tokens.pop_back();
    }

    // Add EOS
    tokens.push_back(tok->eos_id);

    return tokens;
}

//
// Tensor mapping helpers
//

static void map_encoder_layer_tensor(const char * name, ggml_tensor * t, magpie_encoder_layer & layer) {
    // Actual patterns from GGUF:
    // encoder.layers.N.norm_self.weight
    // encoder.layers.N.self_attention.qkv_net.weight
    // encoder.layers.N.self_attention.o_net.weight
    // encoder.layers.N.norm_pos_ff.weight
    // encoder.layers.N.pos_ff.proj.conv.weight
    // encoder.layers.N.pos_ff.o_net.conv.weight

    if (strstr(name, "norm_self.weight")) {
        layer.norm_self_w = t;
    } else if (strstr(name, "self_attention.qkv_net.weight")) {
        layer.sa_qkv_w = t;
    } else if (strstr(name, "self_attention.o_net.weight")) {
        layer.sa_out_w = t;
    } else if (strstr(name, "norm_pos_ff.weight")) {
        layer.norm_ff_w = t;
    } else if (strstr(name, "pos_ff.proj.conv.weight")) {
        layer.ff_proj_w = t;
    } else if (strstr(name, "pos_ff.o_net.conv.weight")) {
        layer.ff_out_w = t;
    }
}

static void map_decoder_layer_tensor(const char * name, ggml_tensor * t, magpie_decoder_layer & layer) {
    // Actual patterns from GGUF:
    // decoder.layers.N.norm_self.weight
    // decoder.layers.N.self_attention.qkv_net.weight
    // decoder.layers.N.self_attention.o_net.weight
    // decoder.layers.N.norm_xattn_query.weight
    // decoder.layers.N.cross_attention.q_net.weight
    // decoder.layers.N.cross_attention.kv_net.weight
    // decoder.layers.N.cross_attention.o_net.weight
    // decoder.layers.N.norm_xattn_memory.weight
    // decoder.layers.N.norm_pos_ff.weight
    // decoder.layers.N.pos_ff.proj.conv.weight
    // decoder.layers.N.pos_ff.o_net.conv.weight

    if (strstr(name, "norm_self.weight")) {
        layer.norm_self_w = t;
    } else if (strstr(name, "self_attention.qkv_net.weight")) {
        layer.sa_qkv_w = t;
    } else if (strstr(name, "self_attention.o_net.weight")) {
        layer.sa_out_w = t;
    } else if (strstr(name, "norm_xattn_query.weight")) {
        layer.norm_xa_q_w = t;
    } else if (strstr(name, "cross_attention.q_net.weight")) {
        layer.xa_q_w = t;
    } else if (strstr(name, "cross_attention.kv_net.weight")) {
        layer.xa_kv_w = t;
    } else if (strstr(name, "cross_attention.o_net.weight")) {
        layer.xa_out_w = t;
    } else if (strstr(name, "norm_xattn_memory.weight")) {
        layer.norm_xa_mem_w = t;
    } else if (strstr(name, "norm_pos_ff.weight")) {
        layer.norm_ff_w = t;
    } else if (strstr(name, "pos_ff.proj.conv.weight")) {
        layer.ff_proj_w = t;
    } else if (strstr(name, "pos_ff.o_net.conv.weight")) {
        layer.ff_out_w = t;
    }
}

static int parse_layer_idx(const char * name, const char * prefix) {
    // Find prefix, then parse number after it
    const char * p = strstr(name, prefix);
    if (!p) return -1;
    p += strlen(prefix);
    return atoi(p);
}

static void create_tensors(gguf_context * gguf_ctx, ggml_context * meta_ctx, magpie_model & model) {
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    const auto & hp = model.hparams;

    // Resize layer vectors
    model.encoder.layers.resize(hp.enc_layers);
    model.decoder.layers.resize(hp.dec_layers);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);

        // Get tensor from meta context (created during gguf init)
        ggml_tensor * meta = ggml_get_tensor(meta_ctx, name);
        if (!meta) {
            fprintf(stderr, "magpie: tensor '%s' not found in meta context\n", name);
            continue;
        }

        // Create tensor in model context with same shape
        ggml_tensor * t = ggml_dup_tensor(model.ctx_w, meta);
        ggml_set_name(t, name);

        // Map to model structure based on name
        // Actual tensor names from NeMo checkpoint:
        // - text_embedding.weight
        // - audio_embeddings.N.weight  (note: plural)
        // - baked_context_embedding.weight
        // - encoder.layers.N.* / encoder.norm.weight / encoder.pos_emb.weight
        // - decoder.layers.N.* / decoder.norm.weight / decoder.pos_emb.weight
        // - final_proj.weight / final_proj.bias
        // - local_transformer_in_projection.weight/bias
        // - local_transformer.layers.0.* (single layer)
        // - local_transformer_out_projections.N.weight/bias
        // - local_transformer_positional_embedding.weight

        if (strcmp(name, "text_embedding.weight") == 0) {
            model.embeddings.text_emb_w = t;
        } else if (strstr(name, "audio_embeddings.")) {
            // audio_embeddings.N.weight
            int cb_idx = parse_layer_idx(name, "audio_embeddings.");
            if (cb_idx >= 0 && cb_idx < 8) {
                model.embeddings.audio_emb_w[cb_idx] = t;
            }
        } else if (strcmp(name, "baked_context_embedding.weight") == 0) {
            model.embeddings.baked_context_w = t;
        } else if (strcmp(name, "encoder.position_embeddings.weight") == 0) {
            model.encoder.pos_emb_w = t;
        } else if (strstr(name, "encoder.layers.")) {
            int layer_idx = parse_layer_idx(name, "encoder.layers.");
            if (layer_idx >= 0 && layer_idx < hp.enc_layers) {
                map_encoder_layer_tensor(name, t, model.encoder.layers[layer_idx]);
            }
        } else if (strcmp(name, "encoder.norm_out.weight") == 0) {
            model.encoder.norm_out_w = t;
        } else if (strcmp(name, "decoder.position_embeddings.weight") == 0) {
            model.decoder.pos_emb_w = t;
        } else if (strstr(name, "decoder.layers.")) {
            int layer_idx = parse_layer_idx(name, "decoder.layers.");
            if (layer_idx >= 0 && layer_idx < hp.dec_layers) {
                map_decoder_layer_tensor(name, t, model.decoder.layers[layer_idx]);
            }
        } else if (strcmp(name, "decoder.norm_out.weight") == 0) {
            model.decoder.norm_out_w = t;
        } else if (strcmp(name, "final_proj.weight") == 0) {
            model.final_proj.weight = t;
        } else if (strcmp(name, "final_proj.bias") == 0) {
            model.final_proj.bias = t;
        } else if (strstr(name, "local_transformer_in_projection.weight")) {
            model.local_transformer.in_proj_w = t;
        } else if (strstr(name, "local_transformer_in_projection.bias")) {
            model.local_transformer.in_proj_b = t;
        } else if (strcmp(name, "local_transformer.position_embeddings.weight") == 0) {
            model.local_transformer.pos_emb_w = t;
        } else if (strstr(name, "local_transformer.layers.0.norm_self.weight")) {
            model.local_transformer.norm_self_w = t;
        } else if (strstr(name, "local_transformer.layers.0.self_attention.qkv_net.weight")) {
            model.local_transformer.sa_qkv_w = t;
        } else if (strstr(name, "local_transformer.layers.0.self_attention.o_net.weight")) {
            model.local_transformer.sa_out_w = t;
        } else if (strstr(name, "local_transformer.layers.0.norm_pos_ff.weight")) {
            model.local_transformer.norm_ff_w = t;
        } else if (strstr(name, "local_transformer.layers.0.pos_ff.proj.conv.weight")) {
            model.local_transformer.ff_proj_w = t;
        } else if (strstr(name, "local_transformer.layers.0.pos_ff.o_net.conv.weight")) {
            model.local_transformer.ff_out_w = t;
        } else if (strstr(name, "local_transformer_out_projections.")) {
            // local_transformer_out_projections.N.weight / bias
            int cb_idx = parse_layer_idx(name, "local_transformer_out_projections.");
            if (cb_idx >= 0 && cb_idx < 8) {
                if (strstr(name, ".weight")) {
                    model.local_transformer.out_proj_w[cb_idx] = t;
                } else if (strstr(name, ".bias")) {
                    model.local_transformer.out_proj_b[cb_idx] = t;
                }
            }
        }

        // Store in tensor map for debugging/inspection
        model.tensors[name] = t;
    }
}

static bool load_tensor_data(const char * path, gguf_context * gguf_ctx, magpie_model & model) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "magpie: failed to open '%s' for reading tensor data\n", path);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);

        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            fprintf(stderr, "magpie: tensor '%s' not mapped\n", name);
            continue;
        }

        ggml_tensor * t = it->second;
        size_t nbytes = ggml_nbytes(t);

        // Seek to tensor data
        if (fseek(f, (long)offset, SEEK_SET) != 0) {
            fprintf(stderr, "magpie: failed to seek to tensor '%s'\n", name);
            fclose(f);
            return false;
        }

        // Read into backend buffer
        void * buf = malloc(nbytes);
        if (fread(buf, 1, nbytes, f) != nbytes) {
            fprintf(stderr, "magpie: failed to read tensor '%s'\n", name);
            free(buf);
            fclose(f);
            return false;
        }

        ggml_backend_tensor_set(t, buf, 0, nbytes);
        free(buf);
    }

    fclose(f);
    return true;
}

static void init_kv_cache(magpie_context * ctx) {
    const auto & hp = ctx->model.hparams;
    int max_seq = hp.max_dec_steps + hp.context_frames + 10;  // margin

    // Calculate memory needed for cache context
    size_t n_cache_tensors = hp.dec_layers * 4;  // k, v, xa_k, xa_v per layer
    size_t cache_ctx_size = ggml_tensor_overhead() * n_cache_tensors + 1024;

    struct ggml_init_params params = {
        .mem_size   = cache_ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ctx->state.kv_cache.ctx = ggml_init(params);
    if (!ctx->state.kv_cache.ctx) {
        fprintf(stderr, "magpie: failed to init KV cache context\n");
        return;
    }

    // Create cache tensors
    for (int l = 0; l < hp.dec_layers; l++) {
        // Self-attention cache
        ctx->state.kv_cache.k_cache.push_back(
            ggml_new_tensor_2d(ctx->state.kv_cache.ctx, GGML_TYPE_F16,
                               hp.dec_sa_heads * hp.d_head, max_seq)
        );
        ctx->state.kv_cache.v_cache.push_back(
            ggml_new_tensor_2d(ctx->state.kv_cache.ctx, GGML_TYPE_F16,
                               hp.dec_sa_heads * hp.d_head, max_seq)
        );

        // Cross-attention cache (uses different head dim)
        ctx->state.kv_cache.xa_k_cache.push_back(
            ggml_new_tensor_2d(ctx->state.kv_cache.ctx, GGML_TYPE_F16,
                               hp.dec_xa_heads * hp.dec_xa_d_head, max_seq)
        );
        ctx->state.kv_cache.xa_v_cache.push_back(
            ggml_new_tensor_2d(ctx->state.kv_cache.ctx, GGML_TYPE_F16,
                               hp.dec_xa_heads * hp.dec_xa_d_head, max_seq)
        );
    }

    // Allocate cache buffer on backend
    ctx->state.kv_cache.buffer = ggml_backend_alloc_ctx_tensors(
        ctx->state.kv_cache.ctx, ctx->model.backend
    );

    ctx->state.kv_cache.max_seq = max_seq;
    ctx->state.kv_cache.seq_len = 0;
    ctx->state.kv_cache.enc_seq_len = 0;
}

//
// Public API
//

magpie_context * magpie_init(const char * model_path) {
    return magpie_init_with_backend(model_path, MAGPIE_BACKEND_AUTO);
}

magpie_context * magpie_init_with_backend(const char * model_path, magpie_backend_type backend) {
    magpie_context * ctx = new magpie_context();

    // 1. Initialize backend
    if (!init_backend(ctx->model, backend)) {
        fprintf(stderr, "magpie: failed to init backend\n");
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "magpie: using backend: %s\n", magpie_get_backend_name(ctx));

    // 2. Open GGUF file and read metadata
    struct gguf_init_params gguf_params = {
        .no_alloc = true,
        .ctx      = nullptr,  // will be set by gguf_init
    };

    ggml_context * meta_ctx = nullptr;
    gguf_params.ctx = &meta_ctx;

    gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "magpie: failed to open '%s'\n", model_path);
        magpie_free(ctx);
        return nullptr;
    }

    // 3. Read hyperparameters
    read_hparams(gguf_ctx, ctx->model.hparams);

    const auto & hp = ctx->model.hparams;
    fprintf(stderr, "magpie: d_model=%d, enc_layers=%d, dec_layers=%d\n",
            hp.d_model, hp.enc_layers, hp.dec_layers);

    // 3b. Initialize tokenizer from GGUF metadata
    if (magpie_tokenizer_init(&ctx->model.tokenizer, gguf_ctx)) {
        fprintf(stderr, "magpie: tokenizer loaded\n");
    } else {
        fprintf(stderr, "magpie: tokenizer not available (text input must be pre-tokenized)\n");
    }

    // 4. Create weight context
    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + 1024 * 1024;

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    ctx->model.ctx_w = ggml_init(params);
    if (!ctx->model.ctx_w) {
        fprintf(stderr, "magpie: failed to init weight context\n");
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        magpie_free(ctx);
        return nullptr;
    }

    // 5. Create tensors and map to model structure
    create_tensors(gguf_ctx, meta_ctx, ctx->model);

    fprintf(stderr, "magpie: loaded %d tensors\n", (int)ctx->model.tensors.size());

    // 6. Allocate backend buffer for all weights
    ctx->model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx->model.ctx_w, ctx->model.backend);
    if (!ctx->model.buffer_w) {
        fprintf(stderr, "magpie: failed to allocate weight buffer\n");
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        magpie_free(ctx);
        return nullptr;
    }

    // 7. Load tensor data from GGUF file
    if (!load_tensor_data(model_path, gguf_ctx, ctx->model)) {
        fprintf(stderr, "magpie: failed to load tensor data\n");
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        magpie_free(ctx);
        return nullptr;
    }

    // 8. Initialize KV cache
    init_kv_cache(ctx);

    // 9. Create graph allocator
    ctx->state.allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->model.backend)
    );

    // Clean up GGUF context (no longer needed)
    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);

    fprintf(stderr, "magpie: model loaded successfully\n");
    return ctx;
}

void magpie_free(magpie_context * ctx) {
    if (!ctx) return;

    if (ctx->state.allocr) {
        ggml_gallocr_free(ctx->state.allocr);
    }

    if (ctx->state.kv_cache.buffer) {
        ggml_backend_buffer_free(ctx->state.kv_cache.buffer);
    }

    if (ctx->state.kv_cache.ctx) {
        ggml_free(ctx->state.kv_cache.ctx);
    }

    if (ctx->model.buffer_w) {
        ggml_backend_buffer_free(ctx->model.buffer_w);
    }

    if (ctx->model.ctx_w) {
        ggml_free(ctx->model.ctx_w);
    }

    if (ctx->model.backend) {
        ggml_backend_free(ctx->model.backend);
    }

    delete ctx;
}

const char * magpie_get_backend_name(magpie_context * ctx) {
    if (!ctx || !ctx->model.backend) return "none";
    return ggml_backend_name(ctx->model.backend);
}

bool magpie_model_load(const std::string & path, magpie_model & model, magpie_backend_type backend) {
    // This function is a simplified version that just loads into an existing model
    // For now, use magpie_init_with_backend instead
    (void)path;
    (void)model;
    (void)backend;
    return false;  // Not implemented yet
}

//
// Stub implementations for other functions (to be implemented)
//

ggml_cgraph * magpie_build_encoder_graph(
    magpie_context * ctx,
    ggml_context * ctx0,
    ggml_tensor * text_tokens) {
    (void)ctx; (void)ctx0; (void)text_tokens;
    return nullptr;  // TODO
}

ggml_cgraph * magpie_build_decoder_step_graph(
    magpie_context * ctx,
    ggml_context * ctx0) {
    (void)ctx; (void)ctx0;
    return nullptr;  // TODO
}

// Helper: Build local transformer layer (single transformer layer)
static ggml_tensor * build_local_transformer_layer(
    ggml_context * ctx,
    ggml_tensor * input,        // [lt_dim, seq_len]
    magpie_local_transformer * lt,
    const magpie_hparams * hparams) {

    if (!input || !lt) return nullptr;

    const int64_t lt_dim = input->ne[0];
    const int64_t seq_len = input->ne[1];

    // Self-attention block
    struct ggml_tensor * residual = input;
    struct ggml_tensor * x = magpie_build_layer_norm(ctx, input, lt->norm_self_w, hparams->eps);

    // For local transformer, we use causal self-attention
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(mask, "lt_causal_mask");
    ggml_set_input(mask);

    x = magpie_build_self_attention_with_mask(ctx, x, lt->sa_qkv_w, lt->sa_out_w,
                                               hparams->lt_heads, true, mask);
    x = ggml_add(ctx, x, residual);

    // FFN block (kernel=1, pointwise)
    residual = x;
    x = magpie_build_layer_norm(ctx, x, lt->norm_ff_w, hparams->eps);
    x = magpie_build_conv_ffn(ctx, x, lt->ff_proj_w, lt->ff_out_w, 1);  // kernel_size=1
    x = ggml_add(ctx, x, residual);

    return x;
}

// Build local transformer for first codebook only (single step, position 0)
ggml_tensor * magpie_build_local_transformer_step0(
    ggml_context * ctx,
    ggml_tensor * decoder_hidden,  // [d_model] (768)
    magpie_local_transformer * lt,
    const magpie_hparams * hparams) {

    if (!decoder_hidden || !lt) return nullptr;

    // Step 1: Input projection with bias
    // decoder_hidden: [d_model] -> [lt_dim]
    struct ggml_tensor * projected = ggml_mul_mat(ctx, lt->in_proj_w, decoder_hidden);
    projected = ggml_add(ctx, projected, lt->in_proj_b);  // [lt_dim]

    // Step 2: Add position embedding for position 0
    // pos_emb_w is [lt_dim, 10] in GGML order
    struct ggml_tensor * pos0 = ggml_view_1d(ctx, lt->pos_emb_w, hparams->lt_dim, 0);
    projected = ggml_add(ctx, projected, pos0);  // [lt_dim]

    // Step 3: Reshape to [lt_dim, 1] for transformer layer
    struct ggml_tensor * seq = ggml_reshape_2d(ctx, projected, hparams->lt_dim, 1);

    // Step 4: Run through transformer layer
    struct ggml_tensor * layer_out = build_local_transformer_layer(ctx, seq, lt, hparams);

    // Step 5: Get logits for codebook 0
    // layer_out: [lt_dim, 1] -> [lt_dim]
    struct ggml_tensor * hidden = ggml_view_1d(ctx, layer_out, hparams->lt_dim, 0);
    struct ggml_tensor * logits = ggml_mul_mat(ctx, lt->out_proj_w[0], hidden);
    logits = ggml_add(ctx, logits, lt->out_proj_b[0]);  // [vocab_per_cb]

    return logits;
}

// Build local transformer for a given sequence of projected embeddings
// seq_input: [lt_dim, seq_len] - sequence of projected embeddings
// Returns: [lt_dim, seq_len] - hidden states after transformer layer
ggml_tensor * magpie_build_local_transformer_seq(
    ggml_context * ctx,
    ggml_tensor * seq_input,      // [lt_dim, seq_len]
    magpie_local_transformer * lt,
    const magpie_hparams * hparams) {

    if (!seq_input || !lt) return nullptr;

    const int64_t lt_dim = seq_input->ne[0];
    const int64_t seq_len = seq_input->ne[1];

    // Add position embeddings for all positions
    // pos_emb_w is [lt_dim, max_pos]
    struct ggml_tensor * pos_slice = ggml_view_2d(ctx, lt->pos_emb_w,
        lt_dim, seq_len, lt->pos_emb_w->nb[1], 0);
    struct ggml_tensor * with_pos = ggml_add(ctx, seq_input, pos_slice);

    // Run through transformer layer
    return build_local_transformer_layer(ctx, with_pos, lt, hparams);
}

// Get logits for a specific codebook from local transformer hidden state
ggml_tensor * magpie_build_lt_output_proj(
    ggml_context * ctx,
    ggml_tensor * hidden,     // [lt_dim] - last position hidden state
    magpie_local_transformer * lt,
    int codebook_idx) {

    if (!hidden || !lt || codebook_idx < 0 || codebook_idx >= 8) return nullptr;

    struct ggml_tensor * logits = ggml_mul_mat(ctx, lt->out_proj_w[codebook_idx], hidden);
    logits = ggml_add(ctx, logits, lt->out_proj_b[codebook_idx]);
    return logits;
}

ggml_tensor * magpie_build_local_transformer(
    magpie_context * ctx,
    ggml_context * ctx0,
    ggml_tensor * decoder_hidden,
    ggml_tensor * prev_codes) {
    // Full local transformer: autoregressively predicts codes for all 8 codebooks
    // This is complex because it requires multiple graph evaluations with sampling
    // For now, we implement a simplified version that returns logits for codebook 0
    //
    // decoder_hidden: [d_model] from decoder output
    // prev_codes: [num_codebooks] previously sampled codes (for non-first positions)
    //
    // Returns: logits for the next codebook

    (void)prev_codes;  // TODO: implement multi-step inference

    return magpie_build_local_transformer_step0(ctx0, decoder_hidden,
                                                 &ctx->model.local_transformer,
                                                 &ctx->model.hparams);
}

// Helper: Apply softmax with temperature, then sample from top-k
static int32_t sample_top_k(const std::vector<float> & logits, float temperature, int top_k, std::mt19937 & rng) {
    const int n = (int)logits.size();

    // Find top-k indices
    std::vector<std::pair<float, int>> scored(n);
    for (int i = 0; i < n; i++) {
        scored[i] = {logits[i], i};
    }

    // Partial sort to get top-k
    int k = std::min(top_k, n);
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    // Apply temperature-scaled softmax to top-k
    std::vector<float> probs(k);
    float max_logit = scored[0].first;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        probs[i] = std::exp((scored[i].first - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < k; i++) {
        probs[i] /= sum;
    }

    // Sample from categorical distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
        cumsum += probs[i];
        if (u < cumsum) {
            return scored[i].second;
        }
    }
    return scored[k - 1].second;  // Fallback
}

// Full local transformer inference - sample all 8 codebooks
// Returns both sampled codes and argmax codes for EOS detection
magpie_sample_result magpie_local_transformer_sample_all(
    magpie_context * mctx,
    const float * decoder_hidden_data,  // [d_model] from decoder output
    float temperature,
    int top_k,
    bool forbid_eos) {

    const auto & hp = mctx->model.hparams;
    auto & lt = mctx->model.local_transformer;
    auto & emb = mctx->model.embeddings;

    magpie_sample_result result;
    result.sampled_codes.resize(8);
    result.argmax_codes.resize(8);

    // Random generator for sampling
    static std::mt19937 rng(std::random_device{}());

    // Forbidden special token indices (all except AUDIO_EOS = 2017):
    // 2016 = AUDIO_BOS, 2018 = CONTEXT_BOS, 2019 = CONTEXT_EOS, 2020 = MASK, 2021-2023 = RESERVED
    std::vector<int> forbidden_tokens = {
        hp.audio_bos_id,  // 2016
        hp.audio_bos_id + 2,  // 2018 = CONTEXT_BOS
        hp.audio_bos_id + 3,  // 2019 = CONTEXT_EOS
        hp.audio_bos_id + 4,  // 2020 = MASK
        hp.audio_bos_id + 5,  // 2021 = RESERVED_1
        hp.audio_bos_id + 6,  // 2022 = RESERVED_2
        hp.audio_bos_id + 7,  // 2023 = RESERVED_3
    };
    // If forbidding EOS, add it to the list
    if (forbid_eos) {
        forbidden_tokens.push_back(hp.audio_eos_id);  // 2017
    }

    // We need to maintain a growing sequence of projected embeddings
    // Start with projected decoder hidden
    std::vector<float> lt_seq_data;
    lt_seq_data.reserve(hp.lt_dim * 9);  // Max 9 positions

    // === Step 0: Project decoder hidden ===
    {
        size_t ctx_size = ggml_tensor_overhead() * 16 + 4 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hp.d_model);
        ggml_set_name(input, "input");
        ggml_set_input(input);

        struct ggml_tensor * output = ggml_mul_mat(ctx0, lt.in_proj_w, input);
        output = ggml_add(ctx0, output, lt.in_proj_b);
        ggml_set_name(output, "output");
        ggml_set_output(output);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, output);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(input, decoder_hidden_data, 0, hp.d_model * sizeof(float));
        ggml_backend_graph_compute(mctx->model.backend, gf);

        std::vector<float> proj_result(hp.lt_dim);
        ggml_backend_tensor_get(output, proj_result.data(), 0, hp.lt_dim * sizeof(float));

        // Add to sequence
        lt_seq_data.insert(lt_seq_data.end(), proj_result.begin(), proj_result.end());

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }

    // === Iterate through codebooks ===
    for (int cb = 0; cb < 8; cb++) {
        int seq_len = cb + 1;  // 1 for cb=0, 2 for cb=1, etc.

        // Build graph for this step
        size_t ctx_size = ggml_tensor_overhead() * 256 + 64 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        // Input is the current sequence [lt_dim, seq_len]
        struct ggml_tensor * seq_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.lt_dim, seq_len);
        ggml_set_name(seq_input, "seq_input");
        ggml_set_input(seq_input);

        // Run transformer layer
        struct ggml_tensor * hidden = magpie_build_local_transformer_seq(ctx0, seq_input, &lt, &hp);

        // Get last position hidden state [lt_dim]
        struct ggml_tensor * last_hidden = ggml_view_1d(ctx0, hidden, hp.lt_dim,
            (seq_len - 1) * hp.lt_dim * sizeof(float));
        last_hidden = ggml_cont(ctx0, last_hidden);

        // Output projection for this codebook
        struct ggml_tensor * logits = magpie_build_lt_output_proj(ctx0, last_hidden, &lt, cb);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
        ggml_build_forward_expand(gf, logits);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        // Set sequence input
        ggml_backend_tensor_set(seq_input, lt_seq_data.data(), 0, hp.lt_dim * seq_len * sizeof(float));

        // Set causal mask
        struct ggml_tensor * mask = ggml_get_tensor(ctx0, "lt_causal_mask");
        if (mask) {
            std::vector<float> mask_data(seq_len * seq_len);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    mask_data[i * seq_len + j] = (j <= i) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
        }

        // Compute
        ggml_backend_graph_compute(mctx->model.backend, gf);

        // Get logits
        std::vector<float> logits_data(hp.vocab_per_cb);
        ggml_backend_tensor_get(logits, logits_data.data(), 0, hp.vocab_per_cb * sizeof(float));

        // Mask forbidden tokens (set to -inf before any processing)
        for (int tok : forbidden_tokens) {
            if (tok >= 0 && tok < hp.vocab_per_cb) {
                logits_data[tok] = -INFINITY;
            }
        }

        // Find argmax (for EOS detection, always computed)
        int argmax = 0;
        float max_val = logits_data[0];
        for (int i = 1; i < hp.vocab_per_cb; i++) {
            if (logits_data[i] > max_val) {
                max_val = logits_data[i];
                argmax = i;
            }
        }
        result.argmax_codes[cb] = argmax;

        // Sample with temperature (or use argmax if temperature is very low)
        int sampled;
        if (temperature < 0.01f) {
            sampled = argmax;
        } else {
            sampled = sample_top_k(logits_data, temperature, top_k, rng);
        }
        result.sampled_codes[cb] = sampled;

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);

        // If not last codebook, embed the sampled code and project to lt_dim
        if (cb < 7) {
            // Embed using audio_embeddings[cb]
            size_t ctx_size2 = ggml_tensor_overhead() * 16 + 4 * 1024 * 1024;
            struct ggml_init_params params2 = { ctx_size2, nullptr, true };
            struct ggml_context * ctx1 = ggml_init(params2);

            struct ggml_tensor * code_idx = ggml_new_tensor_1d(ctx1, GGML_TYPE_I32, 1);
            ggml_set_name(code_idx, "code_idx");
            ggml_set_input(code_idx);

            // Lookup embedding
            struct ggml_tensor * code_emb = ggml_get_rows(ctx1, emb.audio_emb_w[cb], code_idx);
            // code_emb is [d_model, 1], we want [d_model]
            code_emb = ggml_reshape_1d(ctx1, code_emb, hp.d_model);

            // Project to lt_dim
            struct ggml_tensor * code_proj = ggml_mul_mat(ctx1, lt.in_proj_w, code_emb);
            code_proj = ggml_add(ctx1, code_proj, lt.in_proj_b);
            ggml_set_name(code_proj, "code_proj");
            ggml_set_output(code_proj);

            struct ggml_cgraph * gf2 = ggml_new_graph(ctx1);
            ggml_build_forward_expand(gf2, code_proj);

            ggml_gallocr_t allocr2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
            ggml_gallocr_reserve(allocr2, gf2);
            ggml_gallocr_alloc_graph(allocr2, gf2);

            ggml_backend_tensor_set(code_idx, &sampled, 0, sizeof(int32_t));
            ggml_backend_graph_compute(mctx->model.backend, gf2);

            std::vector<float> proj_result(hp.lt_dim);
            ggml_backend_tensor_get(code_proj, proj_result.data(), 0, hp.lt_dim * sizeof(float));

            // Append to sequence
            lt_seq_data.insert(lt_seq_data.end(), proj_result.begin(), proj_result.end());

            ggml_gallocr_free(allocr2);
            ggml_free(ctx1);
        }
    }

    return result;
}

ggml_tensor * magpie_build_text_embedding(
    ggml_context * ctx,
    ggml_tensor * tokens,
    magpie_embeddings * embeddings) {
    // tokens: [n_tokens] (int32)
    // text_emb_w: [d_model, vocab_size]
    // result: [d_model, n_tokens]
    //
    // ggml_get_rows does row selection from embedding matrix:
    //   a: [n_embd, *]
    //   b: [n_rows, ...]
    //   result: [n_embd, n_rows, ...]

    if (!embeddings || !embeddings->text_emb_w) {
        return nullptr;
    }

    struct ggml_tensor * embedded = ggml_get_rows(ctx, embeddings->text_emb_w, tokens);

    return embedded;
}

ggml_tensor * magpie_build_audio_embedding(
    ggml_context * ctx,
    ggml_tensor * codes,
    magpie_embeddings * embeddings) {
    // codes: [num_codebooks] for single frame, or [num_codebooks, seq] for sequence
    // audio_emb_w[cb]: [d_model, vocab_per_cb]
    // result: [d_model, seq] (or [d_model] for single frame)
    //
    // Sum embeddings from all 8 codebooks

    if (!embeddings) return nullptr;

    // Check that all codebook embeddings are loaded
    for (int cb = 0; cb < 8; cb++) {
        if (!embeddings->audio_emb_w[cb]) {
            fprintf(stderr, "magpie: audio_emb_w[%d] not loaded\n", cb);
            return nullptr;
        }
    }

    // codes tensor has shape:
    // - [8] for single frame (1D)
    // - [8, seq] for sequence (2D)
    // We need to extract each codebook's codes and look up embeddings

    const int64_t n_codebooks = codes->ne[0];  // should be 8
    // Check if 2D by seeing if ne[1] > 1 (GGML pads unused dims to 1)
    const int64_t seq_len = codes->ne[1] > 1 ? codes->ne[1] : 1;

    if (n_codebooks != 8) {
        fprintf(stderr, "magpie: expected 8 codebooks, got %lld\n", (long long)n_codebooks);
        return nullptr;
    }

    // For each codebook, extract its codes and look up embeddings
    // codes layout in memory: [cb0_t0, cb1_t0, ..., cb7_t0, cb0_t1, ...]
    // We need to extract [cb_i_t0, cb_i_t1, ...] for each cb_i

    ggml_tensor * sum = nullptr;

    for (int cb = 0; cb < 8; cb++) {
        // Extract codes for this codebook
        // Use ggml_view_1d to get a view of codes for codebook cb
        // Stride is n_codebooks (8) between consecutive tokens for this codebook

        ggml_tensor * cb_codes;
        if (seq_len == 1) {
            // Single frame: just get element at index cb
            // View 1 element starting at offset cb
            cb_codes = ggml_view_1d(ctx, codes, 1, cb * sizeof(int32_t));
        } else {
            // Sequence: need to extract strided elements
            // codes is [8, seq], so codes for cb are at offsets cb, cb+8, cb+16, ...
            // Use ggml_view_2d then reshape, or use gather
            // Actually for contiguous column, ggml stores in column-major order
            // So codes[cb, :] is contiguous if codes is [8, seq]
            // ne[0]=8, ne[1]=seq, stride nb[0]=sizeof(int32), nb[1]=8*sizeof(int32)
            // To get row cb: view at offset cb*sizeof(int32), with shape [seq], stride 8*sizeof(int32)

            // Use ggml_view_1d with proper stride set via ggml_view
            // Actually, let's use ggml_get_rows on a transposed view or similar
            // Simpler approach: reshape codes to access properly

            // For GGML, codes with ne[0]=8, ne[1]=seq has elements at:
            // codes[i,j] at offset (i + j*8) * sizeof(int32)
            // So codes[cb, :] = elements at cb, cb+8, cb+16, ...
            // This is not contiguous, need strided view

            // Create strided 1D view: length=seq_len, offset=cb*sizeof(int32), stride=8*sizeof(int32)
            cb_codes = ggml_view_1d(ctx, codes, seq_len, cb * sizeof(int32_t));
            // Note: ggml_view_1d doesn't set custom stride, need ggml_set_1d or use view_2d

            // Actually for this case, let's use a different approach:
            // Transpose codes from [8, seq] to [seq, 8], then use ggml_get_rows or view
            // Or, since we're summing all codebooks, we can iterate differently
            // For now, assume single frame case works and we'll handle seq>1 later
            fprintf(stderr, "magpie: audio embedding with seq>1 not yet implemented\n");
            return nullptr;
        }

        // Look up embeddings for this codebook
        ggml_tensor * emb = ggml_get_rows(ctx, embeddings->audio_emb_w[cb], cb_codes);

        // Sum into result
        if (cb == 0) {
            sum = emb;
        } else {
            sum = ggml_add(ctx, sum, emb);
        }
    }

    // Scale by 1/(num_codebooks * frame_stacking_factor)
    // For Magpie: num_codebooks=8, frame_stacking_factor=1, so divide by 8
    sum = ggml_scale(ctx, sum, 1.0f / 8.0f);

    return sum;
}

ggml_tensor * magpie_build_add_position_embeddings(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * pos_emb_w,
    int offset) {
    // input: [d_model, seq]
    // pos_emb_w: [d_model, max_seq]
    // Slice pos_emb_w to [d_model, seq] starting at offset, then add to input

    if (!input || !pos_emb_w) return nullptr;

    const int64_t d_model = input->ne[0];
    const int64_t seq_len = input->ne[1];

    // Create a view of position embeddings for the current sequence
    // ggml_view_2d(ctx, tensor, ne0, ne1, stride1, offset)
    // stride1 = d_model * sizeof(float) for contiguous rows
    struct ggml_tensor * pos_slice = ggml_view_2d(
        ctx, pos_emb_w,
        d_model, seq_len,
        pos_emb_w->nb[1],  // stride between rows
        offset * pos_emb_w->nb[1]  // byte offset
    );

    // Add position embeddings to input
    return ggml_add(ctx, input, pos_slice);
}

ggml_tensor * magpie_build_self_attention(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * qkv_weight,
    ggml_tensor * out_weight,
    int n_heads,
    bool is_causal) {
    return magpie_build_self_attention_with_mask(ctx, input, qkv_weight, out_weight, n_heads, is_causal, nullptr);
}

ggml_tensor * magpie_build_self_attention_with_mask(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * qkv_weight,
    ggml_tensor * out_weight,
    int n_heads,
    bool is_causal,
    ggml_tensor * shared_mask) {
    // input: [d_model, seq]
    // qkv_weight: [d_model, 3*d_model] in GGUF
    // out_weight: [d_model, d_model]
    // shared_mask: optional pre-created causal mask to share across layers
    // Returns: [d_model, seq]

    if (!input || !qkv_weight || !out_weight) return nullptr;

    const int64_t d_model = input->ne[0];
    const int64_t seq_len = input->ne[1];
    const int64_t d_head = d_model / n_heads;

    // Compute QKV: [3*d_model, seq]
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, input);

    // Split into Q, K, V each [d_model, seq]
    struct ggml_tensor * q = ggml_view_2d(ctx, qkv, d_model, seq_len, qkv->nb[1], 0);
    struct ggml_tensor * k = ggml_view_2d(ctx, qkv, d_model, seq_len, qkv->nb[1], d_model * sizeof(float));
    struct ggml_tensor * v = ggml_view_2d(ctx, qkv, d_model, seq_len, qkv->nb[1], 2 * d_model * sizeof(float));

    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Reshape for multi-head: [d_head, n_heads, seq]
    q = ggml_reshape_3d(ctx, q, d_head, n_heads, seq_len);
    k = ggml_reshape_3d(ctx, k, d_head, n_heads, seq_len);
    v = ggml_reshape_3d(ctx, v, d_head, n_heads, seq_len);

    // Permute Q, K to [d_head, seq, n_heads] for attention computation
    // flash_attn_ext: Q[d_head, n_batch, n_head], K[d_head, n_kv, n_head], V[d_head, n_kv, n_head]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [d_head, seq, n_heads]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);  // [d_head, seq, n_heads]
    v = ggml_permute(ctx, v, 0, 2, 1, 3);  // [d_head, seq, n_heads]

    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // Manual attention instead of flash attention for numerical accuracy
    // Q, K, V are [d_head, seq, n_heads]

    float scale = 1.0f / sqrtf((float)d_head);

    // Compute attention scores: K.T @ Q for each head
    // ggml_mul_mat(K, Q) with [d_head, seq, n_heads] gives [seq, seq, n_heads]
    // result[j, i, h] = sum_d K[d, j, h] * Q[d, i, h]
    // This gives scores[key_pos, query_pos, head]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [seq_j, seq_i, n_heads]

    // Scale scores
    scores = ggml_scale(ctx, scores, scale);

    // Apply causal mask if needed
    if (is_causal) {
        struct ggml_tensor * mask = shared_mask;
        if (!mask) {
            // Create causal mask [seq, seq]
            // mask[j, i] = 0 if j <= i (key j allowed for query i), -inf otherwise
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
            ggml_set_name(mask, "causal_mask");
            ggml_set_input(mask);  // Will be filled at runtime
        }
        // Add mask to scores (broadcasts over n_heads dimension)
        scores = ggml_add(ctx, scores, mask);
    }

    // Softmax over keys dimension (dim 0 = seq_j in [seq_j, seq_i, n_heads])
    scores = ggml_soft_max(ctx, scores);

    // Apply attention to values: scores @ V
    // scores is [seq_j, seq_i, n_heads], V is [d_head, seq_j, n_heads]
    // We want output[d, i, h] = sum_j scores[j, i, h] * V[d, j, h]
    // Permute V to [seq_j, d_head, n_heads] so first dim matches
    struct ggml_tensor * v_perm = ggml_permute(ctx, v, 1, 0, 2, 3);  // [seq_j, d_head, n_heads]
    v_perm = ggml_cont(ctx, v_perm);

    // ggml_mul_mat(v_perm, scores) with [seq_j, d_head, n_heads] and [seq_j, seq_i, n_heads]
    // result[d, i, h] = sum_j v_perm[j, d, h] * scores[j, i, h]
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, v_perm, scores);  // [d_head, seq_i, n_heads]

    // Permute to [d_head, n_heads, seq] then reshape to [d_model, seq]
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);  // [d_head, n_heads, seq]
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, d_model, seq_len);

    // Output projection: out_weight @ attn_out
    struct ggml_tensor * output = ggml_mul_mat(ctx, out_weight, attn_out);

    return output;
}

ggml_tensor * magpie_build_self_attention_cached(
    ggml_context * ctx,
    ggml_tensor * input,       // [d_model, 1] - single step
    ggml_tensor * qkv_weight,  // [3*d_model, d_model]
    ggml_tensor * out_weight,  // [d_model, d_model]
    ggml_tensor * k_cache_in,  // [d_model, cache_len] or nullptr
    ggml_tensor * v_cache_in,  // [d_model, cache_len] or nullptr
    int n_heads,
    ggml_tensor ** k_cache_out,
    ggml_tensor ** v_cache_out) {
    // KV-cached self-attention for autoregressive decoding
    // Only processes the new token, reuses cached K/V from previous steps

    if (!input || !qkv_weight || !out_weight || !k_cache_out || !v_cache_out) return nullptr;

    const int64_t d_model = input->ne[0];
    const int64_t q_len = input->ne[1];  // Should be 1 for single step
    const int64_t cache_len = k_cache_in ? k_cache_in->ne[1] : 0;
    const int64_t kv_len = cache_len + q_len;  // Full K/V sequence length
    const int64_t d_head = d_model / n_heads;

    // Compute QKV for new token: [3*d_model, q_len]
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, input);

    // Split into Q_new, K_new, V_new each [d_model, q_len]
    struct ggml_tensor * q = ggml_view_2d(ctx, qkv, d_model, q_len, qkv->nb[1], 0);
    struct ggml_tensor * k_new = ggml_view_2d(ctx, qkv, d_model, q_len, qkv->nb[1], d_model * sizeof(float));
    struct ggml_tensor * v_new = ggml_view_2d(ctx, qkv, d_model, q_len, qkv->nb[1], 2 * d_model * sizeof(float));

    q = ggml_cont(ctx, q);
    k_new = ggml_cont(ctx, k_new);
    v_new = ggml_cont(ctx, v_new);

    // Concatenate with cache: K = [k_cache; k_new], V = [v_cache; v_new]
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    if (k_cache_in != nullptr && cache_len > 0) {
        k = ggml_concat(ctx, k_cache_in, k_new, 1);  // [d_model, kv_len]
        v = ggml_concat(ctx, v_cache_in, v_new, 1);  // [d_model, kv_len]
    } else {
        k = k_new;
        v = v_new;
    }

    // Output updated cache
    *k_cache_out = ggml_cont(ctx, k);
    *v_cache_out = ggml_cont(ctx, v);

    // Reshape for multi-head attention
    // Q: [d_head, n_heads, q_len], K/V: [d_head, n_heads, kv_len]
    q = ggml_reshape_3d(ctx, q, d_head, n_heads, q_len);
    k = ggml_reshape_3d(ctx, k, d_head, n_heads, kv_len);
    v = ggml_reshape_3d(ctx, v, d_head, n_heads, kv_len);

    // Permute to [d_head, seq, n_heads] for attention computation
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [d_head, q_len, n_heads]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [d_head, kv_len, n_heads]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [d_head, kv_len, n_heads]

    float scale = 1.0f / sqrtf((float)d_head);

    // Compute attention scores: K.T @ Q -> [kv_len, q_len, n_heads]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, scale);

    // No masking needed - causal is implicit since we only have past K/V
    // (the new token can attend to all cached tokens + itself)

    // Softmax over keys dimension
    scores = ggml_soft_max(ctx, scores);

    // Apply attention to values: V @ scores
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [kv_len, d_head, n_heads]
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, v_perm, scores);  // [d_head, q_len, n_heads]

    // Reshape back to [d_model, q_len]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));  // [d_head, n_heads, q_len]
    attn_out = ggml_reshape_2d(ctx, attn_out, d_model, q_len);

    // Output projection
    struct ggml_tensor * output = ggml_mul_mat(ctx, out_weight, attn_out);

    return output;
}

void magpie_precompute_cross_attention_kv(
    ggml_context * ctx,
    ggml_tensor * encoder_out,  // [d_model, enc_seq]
    ggml_tensor * kv_weight,    // [d_model, 2 * d_xa_head * n_xa_heads] in GGML (transposed)
    ggml_tensor * norm_mem_w,   // [d_model] memory norm weight
    float eps,
    ggml_tensor ** k_out,
    ggml_tensor ** v_out) {
    // Precompute K/V from encoder output for cross-attention
    // This only needs to be done once per utterance

    if (!encoder_out || !kv_weight || !norm_mem_w || !k_out || !v_out) return;

    // kv_weight is [d_model, d_kv] = [768, 256] in GGML layout (transposed)
    // ne[0] = d_model, ne[1] = d_kv
    const int64_t enc_seq = encoder_out->ne[1];
    const int64_t d_kv = kv_weight->ne[1];  // 2 * d_xa_head * n_xa_heads (stored in ne[1])
    const int64_t d_kv_half = d_kv / 2;     // d_xa_head * n_xa_heads

    // Normalize encoder output
    struct ggml_tensor * norm_mem = ggml_norm(ctx, encoder_out, eps);
    norm_mem = ggml_mul(ctx, norm_mem, norm_mem_w);

    // Project to K/V: [d_kv, enc_seq]
    // ggml_mul_mat(A, B) where A is [k, n] and B is [k, m] gives [n, m]
    // kv_weight is [768, 256], norm_mem is [768, 14] -> result is [256, 14]
    struct ggml_tensor * kv = ggml_mul_mat(ctx, kv_weight, norm_mem);
    kv = ggml_cont(ctx, kv);  // Ensure contiguous

    // kv is [256, 14] = [d_kv, enc_seq]
    // We want K = kv[0:128, :] and V = kv[128:256, :]

    // Reshape to [d_kv_half, 2, enc_seq] = [128, 2, 14]
    struct ggml_tensor * kv_3d = ggml_reshape_3d(ctx, kv, d_kv_half, 2, enc_seq);

    // K is kv_3d[:, 0, :], V is kv_3d[:, 1, :]
    *k_out = ggml_cont(ctx, ggml_view_3d(ctx, kv_3d,
        d_kv_half, 1, enc_seq,
        kv_3d->nb[1], kv_3d->nb[2],
        0));  // offset 0 for K
    *v_out = ggml_cont(ctx, ggml_view_3d(ctx, kv_3d,
        d_kv_half, 1, enc_seq,
        kv_3d->nb[1], kv_3d->nb[2],
        kv_3d->nb[1]));  // offset by one slice for V

    // Reshape back to 2D: [d_kv_half, enc_seq]
    *k_out = ggml_reshape_2d(ctx, *k_out, d_kv_half, enc_seq);
    *v_out = ggml_reshape_2d(ctx, *v_out, d_kv_half, enc_seq);
}

ggml_tensor * magpie_build_cross_attention_cached(
    ggml_context * ctx,
    ggml_tensor * query,       // [d_model, 1] - decoder current step
    ggml_tensor * k_cached,    // [d_xa_head * n_xa_heads, enc_seq]
    ggml_tensor * v_cached,    // [d_xa_head * n_xa_heads, enc_seq]
    ggml_tensor * q_weight,    // [d_xa_head * n_xa_heads, d_model]
    ggml_tensor * out_weight,  // [d_model, d_xa_head * n_xa_heads]
    int n_heads,
    int d_head) {
    // Cross-attention with pre-cached K/V from encoder
    // Query is from decoder (current step only)
    // K/V are pre-computed from encoder output

    if (!query || !k_cached || !v_cached || !q_weight || !out_weight) return nullptr;

    const int64_t d_model = query->ne[0];
    const int64_t q_len = query->ne[1];
    const int64_t enc_seq = k_cached->ne[1];
    const int64_t d_qkv = n_heads * d_head;

    // Project query: [d_qkv, q_len]
    struct ggml_tensor * q = ggml_mul_mat(ctx, q_weight, query);

    // Reshape for multi-head attention
    // Q: [d_head, n_heads, q_len], K/V: [d_head, n_heads, enc_seq]
    q = ggml_reshape_3d(ctx, q, d_head, n_heads, q_len);
    struct ggml_tensor * k = ggml_reshape_3d(ctx, k_cached, d_head, n_heads, enc_seq);
    struct ggml_tensor * v = ggml_reshape_3d(ctx, v_cached, d_head, n_heads, enc_seq);

    // Permute to [d_head, seq, n_heads]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [d_head, q_len, n_heads]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [d_head, enc_seq, n_heads]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [d_head, enc_seq, n_heads]

    float scale = 1.0f / sqrtf((float)d_head);

    // Attention scores: K.T @ Q -> [enc_seq, q_len, n_heads]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, scale);

    // No masking for cross-attention (full attention to encoder)
    scores = ggml_soft_max(ctx, scores);

    // Apply to values: V @ scores
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, v_perm, scores);  // [d_head, q_len, n_heads]

    // Reshape back to [d_qkv, q_len]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    attn_out = ggml_reshape_2d(ctx, attn_out, d_qkv, q_len);

    // Output projection
    struct ggml_tensor * output = ggml_mul_mat(ctx, out_weight, attn_out);

    return output;
}

ggml_tensor * magpie_build_conv_ffn(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * proj_weight,
    ggml_tensor * out_weight,
    int kernel_size) {
    // input: [d_model, seq]
    // proj_weight: [kernel, d_model, d_ffn] in GGUF storage order (PyTorch [d_ffn, d_model, kernel])
    // out_weight: [kernel, d_ffn, d_model] in GGUF storage order (PyTorch [d_model, d_ffn, kernel])
    // Returns: [d_model, seq]
    //
    // NeMo FFN: causal Conv1d + GELU + causal Conv1d
    // Causal padding for kernel_size=3: pad (2, 0) on left

    if (!input || !proj_weight || !out_weight) return nullptr;

    const int64_t d_model = input->ne[0];
    const int64_t seq_len = input->ne[1];
    const int64_t d_ffn = proj_weight->ne[2];

    struct ggml_tensor * hidden;

    if (kernel_size == 1) {
        // Pointwise convolution = linear layer
        struct ggml_tensor * proj_w_2d = ggml_view_2d(ctx, proj_weight, d_model, d_ffn,
                                                      proj_weight->nb[2], 0);
        proj_w_2d = ggml_cont(ctx, proj_w_2d);
        hidden = ggml_mul_mat(ctx, proj_w_2d, input);  // [d_ffn, seq]

        // GELU activation
        hidden = ggml_gelu(ctx, hidden);

        // Output projection
        struct ggml_tensor * out_w_2d = ggml_view_2d(ctx, out_weight, d_ffn, d_model,
                                                     out_weight->nb[2], 0);
        out_w_2d = ggml_cont(ctx, out_w_2d);
        return ggml_mul_mat(ctx, out_w_2d, hidden);  // [d_model, seq]
    } else {
        // Causal 1D convolution with kernel_size > 1
        // For causal conv: output[t] = sum over k of weight[k] @ input[t - (kernel_size-1) + k]
        // This means we pad (kernel_size-1) zeros on the left
        //
        // proj_weight: [kernel, d_model, d_ffn] in GGUF storage order (kernel NOT contiguous in row)
        // out_weight: [kernel, d_ffn, d_model] in GGUF storage order
        //
        // We'll permute the weights and use im2col-style unrolling for the convolution.

        int64_t pad_left = kernel_size - 1;  // Causal padding

        // === First projection: d_model -> d_ffn ===
        // proj_weight is [kernel=3, d_model=768, d_ffn=3072]
        // We want to permute to [d_model, d_ffn, kernel] = [768, 3072, 3]
        // Empirically found: ggml_permute(src, 2, 0, 1, 3) gives [768, 3072, 3]
        struct ggml_tensor * proj_perm = ggml_cont(ctx, ggml_permute(ctx, proj_weight, 2, 0, 1, 3));

        // Pad input on the left with zeros: [d_model, seq] -> [d_model, seq + pad_left]
        struct ggml_tensor * padded = ggml_pad_ext(ctx, input, 0, 0, pad_left, 0, 0, 0, 0, 0);

        // Create input views for each kernel position and stack them for batched matmul
        // Then sum the results
        // For kernel_size = 3, we compute:
        //   hidden = W[0] @ pad[:, 0:seq] + W[1] @ pad[:, 1:seq+1] + W[2] @ pad[:, 2:seq+2]

        struct ggml_tensor * term0 = nullptr;
        struct ggml_tensor * term1 = nullptr;
        struct ggml_tensor * term2 = nullptr;

        // k=0
        {
            struct ggml_tensor * input_k = ggml_view_2d(ctx, padded, d_model, seq_len,
                padded->nb[1], 0 * padded->nb[1]);
            struct ggml_tensor * w_k = ggml_view_2d(ctx, proj_perm, d_model, d_ffn,
                proj_perm->nb[1], 0 * proj_perm->nb[2]);
            term0 = ggml_mul_mat(ctx, w_k, input_k);
        }

        // k=1
        if (kernel_size > 1) {
            struct ggml_tensor * input_k = ggml_view_2d(ctx, padded, d_model, seq_len,
                padded->nb[1], 1 * padded->nb[1]);
            struct ggml_tensor * w_k = ggml_view_2d(ctx, proj_perm, d_model, d_ffn,
                proj_perm->nb[1], 1 * proj_perm->nb[2]);
            term1 = ggml_mul_mat(ctx, w_k, input_k);
        }

        // k=2
        if (kernel_size > 2) {
            struct ggml_tensor * input_k = ggml_view_2d(ctx, padded, d_model, seq_len,
                padded->nb[1], 2 * padded->nb[1]);
            struct ggml_tensor * w_k = ggml_view_2d(ctx, proj_perm, d_model, d_ffn,
                proj_perm->nb[1], 2 * proj_perm->nb[2]);
            term2 = ggml_mul_mat(ctx, w_k, input_k);
        }

        // Sum terms
        hidden = term0;
        if (term1) hidden = ggml_add(ctx, hidden, term1);
        if (term2) hidden = ggml_add(ctx, hidden, term2);

        // GELU activation
        hidden = ggml_gelu(ctx, hidden);  // [d_ffn, seq_len]

        // === Second projection: d_ffn -> d_model ===
        // out_weight is [kernel=3, d_ffn=3072, d_model=768]
        // We want to permute to [d_ffn, d_model, kernel] = [3072, 768, 3]
        // Empirically: ggml_permute(src, 2, 0, 1, 3) produces the correct shape
        struct ggml_tensor * out_perm = ggml_cont(ctx, ggml_permute(ctx, out_weight, 2, 0, 1, 3));

        // Pad hidden on the left: [d_ffn, seq] -> [d_ffn, seq + pad_left]
        struct ggml_tensor * hidden_padded = ggml_pad_ext(ctx, hidden, 0, 0, pad_left, 0, 0, 0, 0, 0);

        struct ggml_tensor * out0 = nullptr;
        struct ggml_tensor * out1 = nullptr;
        struct ggml_tensor * out2 = nullptr;

        // k=0
        {
            struct ggml_tensor * hidden_k = ggml_view_2d(ctx, hidden_padded, d_ffn, seq_len,
                hidden_padded->nb[1], 0 * hidden_padded->nb[1]);
            struct ggml_tensor * w_k = ggml_view_2d(ctx, out_perm, d_ffn, d_model,
                out_perm->nb[1], 0 * out_perm->nb[2]);
            out0 = ggml_mul_mat(ctx, w_k, hidden_k);
        }

        // k=1
        if (kernel_size > 1) {
            struct ggml_tensor * hidden_k = ggml_view_2d(ctx, hidden_padded, d_ffn, seq_len,
                hidden_padded->nb[1], 1 * hidden_padded->nb[1]);
            struct ggml_tensor * w_k = ggml_view_2d(ctx, out_perm, d_ffn, d_model,
                out_perm->nb[1], 1 * out_perm->nb[2]);
            out1 = ggml_mul_mat(ctx, w_k, hidden_k);
        }

        // k=2
        if (kernel_size > 2) {
            struct ggml_tensor * hidden_k = ggml_view_2d(ctx, hidden_padded, d_ffn, seq_len,
                hidden_padded->nb[1], 2 * hidden_padded->nb[1]);
            struct ggml_tensor * w_k = ggml_view_2d(ctx, out_perm, d_ffn, d_model,
                out_perm->nb[1], 2 * out_perm->nb[2]);
            out2 = ggml_mul_mat(ctx, w_k, hidden_k);
        }

        // Sum terms
        struct ggml_tensor * output = out0;
        if (out1) output = ggml_add(ctx, output, out1);
        if (out2) output = ggml_add(ctx, output, out2);

        return output;  // [d_model, seq]
    }
}

ggml_tensor * magpie_build_encoder_layer(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * pos_emb,
    magpie_encoder_layer * layer,
    const magpie_hparams * hparams) {
    return magpie_build_encoder_layer_with_mask(ctx, input, pos_emb, layer, hparams, nullptr);
}

ggml_tensor * magpie_build_encoder_layer_with_mask(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * pos_emb,
    magpie_encoder_layer * layer,
    const magpie_hparams * hparams,
    ggml_tensor * shared_mask) {
    (void)pos_emb;  // Position embeddings are added before layer stack

    if (!input || !layer || !hparams) return nullptr;

    // Encoder layer structure:
    // 1. norm_self -> self_attention -> residual
    // 2. norm_ff -> conv_ffn -> residual

    // Self-attention block
    struct ggml_tensor * residual = input;
    struct ggml_tensor * x = magpie_build_layer_norm(ctx, input, layer->norm_self_w, hparams->eps);
    x = magpie_build_self_attention_with_mask(ctx, x, layer->sa_qkv_w, layer->sa_out_w,
                                    hparams->enc_heads, true, shared_mask);  // NeMo encoder uses causal attention
    x = ggml_add(ctx, x, residual);

    // FFN block
    residual = x;
    x = magpie_build_layer_norm(ctx, x, layer->norm_ff_w, hparams->eps);
    x = magpie_build_conv_ffn(ctx, x, layer->ff_proj_w, layer->ff_out_w, hparams->enc_kernel);
    x = ggml_add(ctx, x, residual);

    return x;
}

ggml_tensor * magpie_build_full_encoder(
    ggml_context * ctx,
    ggml_tensor * input,
    magpie_encoder * encoder,
    const magpie_hparams * hparams) {
    // Full encoder: text embedding + pos embeddings + 6 layers + final norm
    // input: [d_model, seq] (already embedded)
    // output: [d_model, seq]

    if (!input || !encoder || !hparams) return nullptr;

    const int64_t seq_len = input->ne[1];

    // 1. Add position embeddings
    struct ggml_tensor * x = magpie_build_add_position_embeddings(
        ctx, input, encoder->pos_emb_w, 0);

    // 2. Create shared causal mask for all layers (F32 to match attention scores)
    struct ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    // 3. Process through all encoder layers
    for (int l = 0; l < hparams->enc_layers; l++) {
        x = magpie_build_encoder_layer_with_mask(ctx, x, nullptr, &encoder->layers[l], hparams, causal_mask);
        if (!x) {
            fprintf(stderr, "magpie: encoder layer %d failed\n", l);
            return nullptr;
        }
    }

    // 4. Apply final norm
    x = magpie_build_layer_norm(ctx, x, encoder->norm_out_w, hparams->eps);

    return x;
}

ggml_tensor * magpie_build_cross_attention(
    ggml_context * ctx,
    ggml_tensor * query,
    ggml_tensor * memory,
    ggml_tensor * q_weight,
    ggml_tensor * kv_weight,
    ggml_tensor * out_weight,
    int n_heads,
    int d_head) {
    // Cross-attention from decoder (query) to encoder (memory)
    // query: [d_model, dec_seq]
    // memory: [d_model, enc_seq]
    // q_weight: [d_xa, d_model] where d_xa = n_heads * d_head
    // kv_weight: [2*d_xa, d_model]
    // out_weight: [d_model, d_xa]
    // Returns: [d_model, dec_seq]

    if (!query || !memory || !q_weight || !kv_weight || !out_weight) return nullptr;

    const int64_t d_model = query->ne[0];
    const int64_t dec_seq = query->ne[1];
    const int64_t enc_seq = memory->ne[1];
    const int64_t d_xa = n_heads * d_head;

    // Compute Q from decoder query: [d_xa, dec_seq]
    struct ggml_tensor * Q = ggml_mul_mat(ctx, q_weight, query);

    // Compute K, V from encoder memory: [2*d_xa, enc_seq]
    struct ggml_tensor * KV = ggml_mul_mat(ctx, kv_weight, memory);

    // Split KV into K and V
    struct ggml_tensor * K = ggml_view_2d(ctx, KV, d_xa, enc_seq, KV->nb[1], 0);
    struct ggml_tensor * V = ggml_view_2d(ctx, KV, d_xa, enc_seq, KV->nb[1], d_xa * sizeof(float));
    K = ggml_cont(ctx, K);
    V = ggml_cont(ctx, V);

    // Reshape for multi-head attention
    // Q: [d_xa, dec_seq] -> [d_head, n_heads, dec_seq]
    // K, V: [d_xa, enc_seq] -> [d_head, n_heads, enc_seq]
    Q = ggml_reshape_3d(ctx, Q, d_head, n_heads, dec_seq);
    K = ggml_reshape_3d(ctx, K, d_head, n_heads, enc_seq);
    V = ggml_reshape_3d(ctx, V, d_head, n_heads, enc_seq);

    // Permute to [d_head, seq, n_heads] for attention computation
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));  // [d_head, dec_seq, n_heads]
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));  // [d_head, enc_seq, n_heads]
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));  // [d_head, enc_seq, n_heads]

    // Attention: Q @ K^T / sqrt(d_head) -> softmax -> @ V
    float scale = 1.0f / sqrtf((float)d_head);

    // Scores: K^T @ Q -> [enc_seq, dec_seq, n_heads]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
    scores = ggml_scale(ctx, scores, scale);

    // Softmax over encoder positions (key dimension = dim 0)
    scores = ggml_soft_max(ctx, scores);

    // Attend to values: V^T @ scores
    // V is [d_head, enc_seq, n_heads]
    // scores is [enc_seq, dec_seq, n_heads]
    // Permute V to [enc_seq, d_head, n_heads]
    struct ggml_tensor * V_perm = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));

    // V_perm @ scores -> [d_head, dec_seq, n_heads]
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, V_perm, scores);

    // Reshape back: [d_head, dec_seq, n_heads] -> [d_head, n_heads, dec_seq] -> [d_xa, dec_seq]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));  // [d_head, n_heads, dec_seq]
    attn_out = ggml_reshape_2d(ctx, attn_out, d_xa, dec_seq);

    // Output projection: [d_model, dec_seq]
    struct ggml_tensor * output = ggml_mul_mat(ctx, out_weight, attn_out);

    return output;
}

// Build decoder layer with optional shared causal mask
static ggml_tensor * magpie_build_decoder_layer_with_mask(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * encoder_out,
    int layer_idx,
    magpie_decoder_layer * layer,
    magpie_kv_cache * kv_cache,
    const magpie_hparams * hparams,
    ggml_tensor * shared_mask) {
    // Decoder layer structure (no KV cache for now - full sequence):
    // 1. norm_self -> self_attention (causal) -> residual
    // 2. norm_xa_query -> cross_attention -> residual
    // 3. norm_ff -> conv_ffn (kernel=1) -> residual

    (void)layer_idx;
    (void)kv_cache;  // KV cache support will be added later

    if (!input || !encoder_out || !layer || !hparams) return nullptr;

    const int64_t seq_len = input->ne[1];

    // === Self-attention block (causal) ===
    struct ggml_tensor * residual = input;
    struct ggml_tensor * x = magpie_build_layer_norm(ctx, input, layer->norm_self_w, hparams->eps);

    // Use shared mask if provided, otherwise create new one
    struct ggml_tensor * sa_mask = shared_mask;
    if (!sa_mask) {
        sa_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
        ggml_set_name(sa_mask, "dec_sa_mask");
        ggml_set_input(sa_mask);
    }

    x = magpie_build_self_attention_with_mask(ctx, x, layer->sa_qkv_w, layer->sa_out_w,
                                               hparams->dec_sa_heads, true, sa_mask);
    x = ggml_add(ctx, x, residual);

    // === Cross-attention block ===
    residual = x;

    // Normalize query (decoder state)
    struct ggml_tensor * norm_q = magpie_build_layer_norm(ctx, x, layer->norm_xa_q_w, hparams->eps);

    // Normalize memory (encoder output)
    struct ggml_tensor * norm_mem = magpie_build_layer_norm(ctx, encoder_out, layer->norm_xa_mem_w, hparams->eps);

    // Cross-attention
    x = magpie_build_cross_attention(ctx, norm_q, norm_mem,
                                      layer->xa_q_w, layer->xa_kv_w, layer->xa_out_w,
                                      hparams->dec_xa_heads, hparams->dec_xa_d_head);
    x = ggml_add(ctx, x, residual);

    // === FFN block (kernel=1, so it's pointwise) ===
    residual = x;
    x = magpie_build_layer_norm(ctx, x, layer->norm_ff_w, hparams->eps);
    x = magpie_build_conv_ffn(ctx, x, layer->ff_proj_w, layer->ff_out_w, hparams->dec_kernel);
    x = ggml_add(ctx, x, residual);

    return x;
}

ggml_tensor * magpie_build_decoder_layer(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * encoder_out,
    int layer_idx,
    magpie_decoder_layer * layer,
    magpie_kv_cache * kv_cache,
    const magpie_hparams * hparams) {
    return magpie_build_decoder_layer_with_mask(ctx, input, encoder_out, layer_idx, layer, kv_cache, hparams, nullptr);
}

ggml_tensor * magpie_build_decoder_layer_cached(
    ggml_context * ctx,
    ggml_tensor * input,           // [d_model, 1] - current step only
    ggml_tensor * xa_k_cached,     // [d_xa_head * n_xa_heads, enc_seq]
    ggml_tensor * xa_v_cached,     // [d_xa_head * n_xa_heads, enc_seq]
    ggml_tensor * sa_k_cache_in,   // [d_model, cache_len] or nullptr
    ggml_tensor * sa_v_cache_in,   // [d_model, cache_len] or nullptr
    int layer_idx,
    magpie_decoder_layer * layer,
    const magpie_hparams * hparams,
    int pos_offset,
    ggml_tensor * pos_emb_w,
    ggml_tensor ** sa_k_cache_out,
    ggml_tensor ** sa_v_cache_out) {
    // Single-step cached decoder layer for autoregressive generation
    // Uses KV cache for self-attention, pre-cached K/V for cross-attention
    //
    // Structure:
    // 1. Add position embedding for current step
    // 2. norm_self -> self_attention_cached -> residual
    // 3. norm_xa_query -> cross_attention_cached -> residual
    // 4. norm_ff -> conv_ffn (kernel=1) -> residual

    (void)layer_idx;

    if (!input || !layer || !hparams || !sa_k_cache_out || !sa_v_cache_out) return nullptr;
    if (!xa_k_cached || !xa_v_cached) return nullptr;

    // Add position embedding for current position
    // input is [d_model, 1], pos_emb_w is [d_model, max_seq]
    struct ggml_tensor * pos_slice = ggml_view_2d(ctx, pos_emb_w,
        hparams->d_model, 1, pos_emb_w->nb[1], pos_offset * pos_emb_w->nb[1]);
    struct ggml_tensor * x = ggml_add(ctx, input, pos_slice);

    // === Self-attention block (cached) ===
    struct ggml_tensor * residual = x;
    x = magpie_build_layer_norm(ctx, x, layer->norm_self_w, hparams->eps);

    // Cached self-attention
    x = magpie_build_self_attention_cached(ctx, x,
        layer->sa_qkv_w, layer->sa_out_w,
        sa_k_cache_in, sa_v_cache_in,
        hparams->dec_sa_heads,
        sa_k_cache_out, sa_v_cache_out);
    if (!x) return nullptr;

    x = ggml_add(ctx, x, residual);

    // === Cross-attention block (cached) ===
    residual = x;

    // Normalize query (decoder state)
    struct ggml_tensor * norm_q = magpie_build_layer_norm(ctx, x, layer->norm_xa_q_w, hparams->eps);

    // Cross-attention with pre-cached K/V from encoder
    x = magpie_build_cross_attention_cached(ctx, norm_q,
        xa_k_cached, xa_v_cached,
        layer->xa_q_w, layer->xa_out_w,
        hparams->dec_xa_heads, hparams->dec_xa_d_head);
    if (!x) return nullptr;

    x = ggml_add(ctx, x, residual);

    // === FFN block (kernel=1, so it's pointwise) ===
    residual = x;
    x = magpie_build_layer_norm(ctx, x, layer->norm_ff_w, hparams->eps);
    x = magpie_build_conv_ffn(ctx, x, layer->ff_proj_w, layer->ff_out_w, hparams->dec_kernel);
    x = ggml_add(ctx, x, residual);

    return x;
}

ggml_tensor * magpie_build_rms_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    float eps) {
    // RMS normalization: x / sqrt(mean(x^2) + eps) * weight
    // input: [d_model, seq] or [d_model]
    // weight: [d_model]
    // output: same shape as input

    if (!input || !weight) return nullptr;

    // GGML has built-in RMS norm
    struct ggml_tensor * norm = ggml_rms_norm(ctx, input, eps);

    // Multiply by weight (element-wise with broadcast)
    return ggml_mul(ctx, norm, weight);
}

ggml_tensor * magpie_build_layer_norm(
    ggml_context * ctx,
    ggml_tensor * input,
    ggml_tensor * weight,
    float eps) {
    // Layer normalization (without bias): (x - mean(x)) / sqrt(var(x) + eps) * weight
    // input: [d_model, seq] or [d_model]
    // weight: [d_model]
    // output: same shape as input
    //
    // Difference from RMSNorm:
    // - LayerNorm subtracts the mean before computing variance
    // - RMSNorm uses root mean square (no mean subtraction)

    if (!input || !weight) return nullptr;

    // GGML has built-in layer norm: ggml_norm()
    // This computes: (x - mean(x)) / sqrt(var(x) + eps)
    struct ggml_tensor * norm = ggml_norm(ctx, input, eps);

    // Multiply by weight (element-wise with broadcast)
    return ggml_mul(ctx, norm, weight);
}

ggml_tensor * magpie_build_final_proj(
    ggml_context * ctx,
    ggml_tensor * input,
    magpie_final_proj * proj) {
    // Final projection: decoder output -> logits for all codebooks
    // input: [d_model, seq] or [d_model] for single frame
    // weight: [num_codebooks * vocab_per_cb, d_model] = [16192, 768]
    // bias: [num_codebooks * vocab_per_cb] = [16192]
    // output: [num_codebooks * vocab_per_cb, seq] or [num_codebooks * vocab_per_cb]

    if (!input || !proj || !proj->weight) return nullptr;

    // Linear projection: W @ input
    struct ggml_tensor * logits = ggml_mul_mat(ctx, proj->weight, input);

    // Add bias if present
    if (proj->bias) {
        logits = ggml_add(ctx, logits, proj->bias);
    }

    return logits;
}

bool magpie_encode_text(
    magpie_context * ctx,
    const int32_t * tokens,
    int n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0) {
        fprintf(stderr, "magpie_encode_text: invalid args\n");
        return false;
    }

    const auto & hp = ctx->model.hparams;

    // Build compute graph for encoder
    size_t ctx_size = ggml_tensor_overhead() * 512 + 128 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "magpie_encode_text: failed to init context\n");
        return false;
    }

    // Create input tensor for tokens
    struct ggml_tensor * input_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(input_tokens, "input_tokens");
    ggml_set_input(input_tokens);

    // Text embedding lookup
    struct ggml_tensor * embedded = magpie_build_text_embedding(ctx0, input_tokens, &ctx->model.embeddings);
    if (!embedded) {
        fprintf(stderr, "magpie_encode_text: text embedding failed\n");
        ggml_free(ctx0);
        return false;
    }

    // Full encoder: embedding + pos + 6 layers + final norm
    struct ggml_tensor * encoded = magpie_build_full_encoder(ctx0, embedded, &ctx->model.encoder, &hp);
    if (!encoded) {
        fprintf(stderr, "magpie_encode_text: encoder failed\n");
        ggml_free(ctx0);
        return false;
    }
    ggml_set_name(encoded, "encoder_output");
    ggml_set_output(encoded);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(gf, encoded);

    // Allocate and compute
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "magpie_encode_text: graph alloc failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        return false;
    }

    // Set input tokens
    ggml_backend_tensor_set(input_tokens, tokens, 0, n_tokens * sizeof(int32_t));

    // Fill causal mask for encoder (NeMo encoder uses causal attention)
    struct ggml_tensor * mask = ggml_get_tensor(ctx0, "causal_mask");
    if (mask) {
        std::vector<float> mask_data(n_tokens * n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            for (int j = 0; j < n_tokens; j++) {
                mask_data[i * n_tokens + j] = (j <= i) ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    // Compute
    if (ggml_backend_graph_compute(ctx->model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "magpie_encode_text: compute failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        return false;
    }

    // Store encoder output in state
    ctx->state.encoder_output.resize(hp.d_model * n_tokens);
    ggml_backend_tensor_get(encoded, ctx->state.encoder_output.data(), 0,
                            ctx->state.encoder_output.size() * sizeof(float));
    ctx->state.enc_seq_len = n_tokens;

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);

    return true;
}

bool magpie_decode_step(
    magpie_context * ctx,
    std::vector<float> & logits) {
    // This function runs one decoder step and returns logits
    // For now, use the full synthesis function which handles the complete loop
    // This is a placeholder for incremental decoding with KV cache
    (void)ctx; (void)logits;
    fprintf(stderr, "magpie_decode_step: use magpie_synthesize_codes for now\n");
    return false;
}

std::vector<int32_t> magpie_sample_frame(
    magpie_context * ctx,
    const std::vector<float> & logits) {
    // Sample codes from logits using local transformer
    // logits should be [num_codebooks * vocab_per_cb] from final_proj
    // This function extracts the first codebook's logits and samples all 8 codes

    if (!ctx || logits.empty()) return {};

    const auto & hp = ctx->model.hparams;

    // For proper sampling, we need the decoder hidden state, not just logits
    // The logits from final_proj are coarse; local transformer refines them
    // This function is a placeholder; use magpie_local_transformer_sample_all instead

    // Simple argmax on each codebook's slice
    std::vector<int32_t> codes(8);
    for (int cb = 0; cb < 8; cb++) {
        int offset = cb * hp.vocab_per_cb;
        float max_val = logits[offset];
        int max_idx = 0;
        for (int i = 1; i < hp.vocab_per_cb; i++) {
            if (logits[offset + i] > max_val) {
                max_val = logits[offset + i];
                max_idx = i;
            }
        }
        codes[cb] = max_idx;
    }

    return codes;
}

// Build full decoder forward pass (without KV cache - full sequence)
static ggml_tensor * build_full_decoder(
    ggml_context * ctx0,
    ggml_tensor * decoder_input,      // [d_model, dec_seq]
    ggml_tensor * encoder_output,     // [d_model, enc_seq]
    magpie_decoder * decoder,
    const magpie_hparams * hparams) {

    if (!decoder_input || !encoder_output || !decoder || !hparams) return nullptr;

    const int64_t dec_seq = decoder_input->ne[1];

    // Add position embeddings
    struct ggml_tensor * x = magpie_build_add_position_embeddings(
        ctx0, decoder_input, decoder->pos_emb_w, 0);

    // Create shared causal mask for self-attention (shared across all layers)
    struct ggml_tensor * sa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dec_seq, dec_seq);
    ggml_set_name(sa_mask, "dec_sa_mask");
    ggml_set_input(sa_mask);

    // Process through all decoder layers with shared mask
    for (int l = 0; l < hparams->dec_layers; l++) {
        x = magpie_build_decoder_layer_with_mask(ctx0, x, encoder_output, l,
                                                  &decoder->layers[l], nullptr, hparams, sa_mask);
        if (!x) {
            fprintf(stderr, "magpie: decoder layer %d failed\n", l);
            return nullptr;
        }
    }

    // Final norm
    x = magpie_build_layer_norm(ctx0, x, decoder->norm_out_w, hparams->eps);

    return x;
}

std::vector<int32_t> magpie_synthesize_codes(
    magpie_context * ctx,
    const int32_t * tokens,
    int n_tokens) {

    if (!ctx || !tokens || n_tokens <= 0) {
        fprintf(stderr, "magpie_synthesize_codes: invalid args\n");
        return {};
    }

    const auto & hp = ctx->model.hparams;

    // Step 1: Encode text
    fprintf(stderr, "magpie: encoding text (%d tokens)...\n", n_tokens);
    if (!magpie_encode_text(ctx, tokens, n_tokens)) {
        fprintf(stderr, "magpie_synthesize_codes: text encoding failed\n");
        return {};
    }
    fprintf(stderr, "magpie: text encoded, output shape [%d, %d]\n", hp.d_model, ctx->state.enc_seq_len);

    // Reset generation state
    ctx->state.generated_codes.clear();
    ctx->state.n_generated_frames = 0;

    // Step 2: Prepare baked context
    // The baked context is [context_frames, d_model] = [110, 768]
    std::vector<float> context_data(hp.context_frames * hp.d_model);

    // Extract baked context for speaker
    {
        size_t ctx_size = ggml_tensor_overhead() * 16 + 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * speaker_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(speaker_idx, "speaker_idx");
        ggml_set_input(speaker_idx);

        // baked_context_w is [d_model * context_frames, num_speakers] after transpose in GGML
        // Actually check the shape
        auto * bcw = ctx->model.embeddings.baked_context_w;
        fprintf(stderr, "magpie: baked_context_w shape [%lld, %lld]\n",
                (long long)bcw->ne[0], (long long)bcw->ne[1]);

        struct ggml_tensor * flat = ggml_get_rows(ctx0, bcw, speaker_idx);
        ggml_set_name(flat, "context_flat");
        ggml_set_output(flat);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, flat);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        int32_t sid = ctx->speaker_id;
        ggml_backend_tensor_set(speaker_idx, &sid, 0, sizeof(int32_t));
        ggml_backend_graph_compute(ctx->model.backend, gf);

        ggml_backend_tensor_get(flat, context_data.data(), 0, context_data.size() * sizeof(float));

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }
    fprintf(stderr, "magpie: baked context loaded for speaker %d\n", ctx->speaker_id);

    // Step 3: Initialize audio sequence with BOS tokens
    std::vector<int32_t> audio_codes;  // [n_frames * 8]
    for (int cb = 0; cb < hp.num_codebooks; cb++) {
        audio_codes.push_back(hp.audio_bos_id);
    }
    ctx->state.n_generated_frames = 1;

    // Step 4: Autoregressive loop
    fprintf(stderr, "magpie: starting autoregressive decoding...\n");

    for (int step = 0; step < hp.max_dec_steps; step++) {
        int n_audio_frames = ctx->state.n_generated_frames;
        int dec_seq = hp.context_frames + n_audio_frames;

        // Build decoder input: [context; audio_embeddings]
        // Compute audio embeddings for all frames
        std::vector<float> decoder_input_data(hp.d_model * dec_seq);

        // Copy context (already in [d_model, context_frames] order for GGML)
        // context_data is flat [context_frames * d_model], needs reshape
        // GGML expects column-major: element [d, t] at index d + t * d_model
        for (int t = 0; t < hp.context_frames; t++) {
            for (int d = 0; d < hp.d_model; d++) {
                // context_data comes from get_rows which gives [first_dim, 1]
                // The baked_context_w is stored as [flat_size, num_speakers]
                // So context_data is flat [d_model * context_frames]
                // PyTorch stores as (T, D), so element [t, d] is at index t * D + d
                // We need GGML [D, T], so element [d, t] is at index d + t * D
                decoder_input_data[d + t * hp.d_model] = context_data[t * hp.d_model + d];
            }
        }

        // Compute audio embeddings for each frame
        {
            size_t ctx_size = ggml_tensor_overhead() * 64 + 32 * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            // For each audio frame, sum embeddings from all codebooks
            for (int t = 0; t < n_audio_frames; t++) {
                // Create tensor for this frame's codes
                struct ggml_tensor * frame_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
                ggml_set_input(frame_codes);

                // Build embedding sum
                struct ggml_tensor * sum = nullptr;
                for (int cb = 0; cb < 8; cb++) {
                    struct ggml_tensor * cb_idx = ggml_view_1d(ctx0, frame_codes, 1, cb * sizeof(int32_t));
                    struct ggml_tensor * emb = ggml_get_rows(ctx0, ctx->model.embeddings.audio_emb_w[cb], cb_idx);
                    emb = ggml_reshape_1d(ctx0, emb, hp.d_model);
                    sum = (sum == nullptr) ? emb : ggml_add(ctx0, sum, emb);
                }
                // Scale by 1/(num_codebooks * frame_stacking_factor) = 1/8
                sum = ggml_scale(ctx0, sum, 1.0f / 8.0f);
                ggml_set_name(sum, "audio_emb");
                ggml_set_output(sum);

                struct ggml_cgraph * gf = ggml_new_graph(ctx0);
                ggml_build_forward_expand(gf, sum);

                ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
                ggml_gallocr_reserve(allocr, gf);
                ggml_gallocr_alloc_graph(allocr, gf);

                // Set frame codes
                std::vector<int32_t> codes(8);
                for (int cb = 0; cb < 8; cb++) {
                    codes[cb] = audio_codes[t * 8 + cb];
                }
                ggml_backend_tensor_set(frame_codes, codes.data(), 0, 8 * sizeof(int32_t));

                ggml_backend_graph_compute(ctx->model.backend, gf);

                // Get embedding and copy to decoder input
                std::vector<float> emb_data(hp.d_model);
                ggml_backend_tensor_get(sum, emb_data.data(), 0, hp.d_model * sizeof(float));

                int dec_t = hp.context_frames + t;
                for (int d = 0; d < hp.d_model; d++) {
                    decoder_input_data[d + dec_t * hp.d_model] = emb_data[d];
                }

                ggml_gallocr_free(allocr);
            }
            ggml_free(ctx0);
        }

        // Run decoder
        std::vector<float> decoder_hidden(hp.d_model);
        {
            // Need more memory as sequence grows - scale with dec_seq
            size_t ctx_size = ggml_tensor_overhead() * 4096 + 1024 * 1024 * 1024;  // 1GB
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            // Create input tensors
            struct ggml_tensor * dec_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_model, dec_seq);
            ggml_set_name(dec_input, "dec_input");
            ggml_set_input(dec_input);

            struct ggml_tensor * enc_output = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_model, ctx->state.enc_seq_len);
            ggml_set_name(enc_output, "enc_output");
            ggml_set_input(enc_output);

            // Build decoder
            struct ggml_tensor * dec_out = build_full_decoder(ctx0, dec_input, enc_output,
                                                               &ctx->model.decoder, &hp);
            if (!dec_out) {
                fprintf(stderr, "magpie: decoder build failed\n");
                ggml_free(ctx0);
                return {};
            }

            // Get last position hidden state
            struct ggml_tensor * last_hidden = ggml_view_1d(ctx0, dec_out, hp.d_model,
                (dec_seq - 1) * hp.d_model * sizeof(float));
            last_hidden = ggml_cont(ctx0, last_hidden);
            ggml_set_name(last_hidden, "last_hidden");
            ggml_set_output(last_hidden);

            struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);
            ggml_build_forward_expand(gf, last_hidden);

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
            if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "magpie: decoder alloc failed\n");
                ggml_gallocr_free(allocr);
                ggml_free(ctx0);
                return {};
            }

            // Set inputs
            ggml_backend_tensor_set(dec_input, decoder_input_data.data(), 0,
                                    decoder_input_data.size() * sizeof(float));
            ggml_backend_tensor_set(enc_output, ctx->state.encoder_output.data(), 0,
                                    ctx->state.encoder_output.size() * sizeof(float));

            // Fill causal mask
            struct ggml_tensor * mask = ggml_get_tensor(ctx0, "dec_sa_mask");
            if (mask) {
                std::vector<float> mask_data(dec_seq * dec_seq);
                for (int i = 0; i < dec_seq; i++) {
                    for (int j = 0; j < dec_seq; j++) {
                        mask_data[i * dec_seq + j] = (j <= i) ? 0.0f : -INFINITY;
                    }
                }
                ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
            }

            if (ggml_backend_graph_compute(ctx->model.backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "magpie: decoder compute failed\n");
                ggml_gallocr_free(allocr);
                ggml_free(ctx0);
                return {};
            }

            ggml_backend_tensor_get(last_hidden, decoder_hidden.data(), 0, hp.d_model * sizeof(float));

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
        }

        // Run local transformer to sample codes for all 8 codebooks
        // Forbid EOS during first min_generated_frames (4)
        const int min_generated_frames = 4;
        bool forbid_eos = (step < min_generated_frames);
        magpie_sample_result sample_result = magpie_local_transformer_sample_all(
            ctx, decoder_hidden.data(), ctx->temperature, ctx->top_k, forbid_eos);

        if (sample_result.sampled_codes.size() != 8) {
            fprintf(stderr, "magpie: local transformer failed\n");
            return {};
        }

        // Debug: print first few frames' codes
        if (step < 5 || (step % 50) == 0) {
            fprintf(stderr, "magpie: step %d codes:", step);
            for (int cb = 0; cb < 8; cb++) {
                fprintf(stderr, " %d", sample_result.sampled_codes[cb]);
            }
            fprintf(stderr, "\n");
        }

        // Check for EOS using argmax_or_multinomial_any method:
        // EOS detected if ANY codebook has EOS in EITHER argmax OR sampled codes
        bool has_eos = false;
        for (int cb = 0; cb < 8; cb++) {
            if (sample_result.sampled_codes[cb] == hp.audio_eos_id ||
                sample_result.argmax_codes[cb] == hp.audio_eos_id) {
                has_eos = true;
                break;
            }
        }

        if (has_eos) {
            fprintf(stderr, "magpie: EOS detected at step %d\n", step);
            break;
        }

        // Add frame codes to sequence (use sampled codes)
        for (int cb = 0; cb < 8; cb++) {
            audio_codes.push_back(sample_result.sampled_codes[cb]);
        }
        ctx->state.n_generated_frames++;

        if ((step + 1) % 10 == 0) {
            fprintf(stderr, "magpie: generated %d frames...\n", ctx->state.n_generated_frames);
        }
    }

    // Return generated codes (excluding BOS frame)
    std::vector<int32_t> result;
    for (size_t i = 8; i < audio_codes.size(); i++) {
        result.push_back(audio_codes[i]);
    }

    fprintf(stderr, "magpie: synthesis complete, %d audio frames generated\n",
            (int)(result.size() / 8));

    return result;
}

// Helper to compute audio embedding for a single frame
static void compute_single_frame_audio_embedding(
    magpie_context * ctx,
    const int32_t * frame_codes,  // [8] codes for this frame
    std::vector<float> & emb_out) {  // [d_model] output

    const auto & hp = ctx->model.hparams;
    emb_out.resize(hp.d_model);

    size_t ctx_size = ggml_tensor_overhead() * 32 + 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
    ggml_set_input(codes);

    // Sum embeddings from all codebooks
    struct ggml_tensor * sum = nullptr;
    for (int cb = 0; cb < 8; cb++) {
        struct ggml_tensor * cb_idx = ggml_view_1d(ctx0, codes, 1, cb * sizeof(int32_t));
        struct ggml_tensor * emb = ggml_get_rows(ctx0, ctx->model.embeddings.audio_emb_w[cb], cb_idx);
        emb = ggml_reshape_1d(ctx0, emb, hp.d_model);
        sum = (sum == nullptr) ? emb : ggml_add(ctx0, sum, emb);
    }
    // Scale by 1/8
    sum = ggml_scale(ctx0, sum, 1.0f / 8.0f);
    ggml_set_name(sum, "audio_emb");
    ggml_set_output(sum);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, sum);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(codes, frame_codes, 0, 8 * sizeof(int32_t));
    ggml_backend_graph_compute(ctx->model.backend, gf);
    ggml_backend_tensor_get(sum, emb_out.data(), 0, hp.d_model * sizeof(float));

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
}

std::vector<int32_t> magpie_synthesize_codes_cached(
    magpie_context * ctx,
    const int32_t * tokens,
    int n_tokens) {

    if (!ctx || !tokens || n_tokens <= 0) {
        fprintf(stderr, "magpie_synthesize_codes_cached: invalid args\n");
        return {};
    }

    const auto & hp = ctx->model.hparams;
    const int d_model = hp.d_model;
    const int n_dec_layers = hp.dec_layers;
    const int d_xa = hp.dec_xa_heads * hp.dec_xa_d_head;

    // Step 1: Encode text
    fprintf(stderr, "magpie: [cached] encoding text (%d tokens)...\n", n_tokens);
    if (!magpie_encode_text(ctx, tokens, n_tokens)) {
        fprintf(stderr, "magpie_synthesize_codes_cached: text encoding failed\n");
        return {};
    }
    const int enc_seq = ctx->state.enc_seq_len;
    fprintf(stderr, "magpie: text encoded, output shape [%d, %d]\n", d_model, enc_seq);

    // Step 2: Extract baked context
    std::vector<float> context_data(hp.context_frames * d_model);
    {
        size_t ctx_size = ggml_tensor_overhead() * 16 + 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * speaker_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(speaker_idx, "speaker_idx");
        ggml_set_input(speaker_idx);

        struct ggml_tensor * flat = ggml_get_rows(ctx0, ctx->model.embeddings.baked_context_w, speaker_idx);
        ggml_set_name(flat, "context_flat");
        ggml_set_output(flat);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, flat);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        int32_t sid = ctx->speaker_id;
        ggml_backend_tensor_set(speaker_idx, &sid, 0, sizeof(int32_t));
        ggml_backend_graph_compute(ctx->model.backend, gf);
        ggml_backend_tensor_get(flat, context_data.data(), 0, context_data.size() * sizeof(float));

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }
    fprintf(stderr, "magpie: baked context loaded for speaker %d\n", ctx->speaker_id);

    // Step 3: Pre-compute cross-attention K/V from encoder output for all layers
    // These are reused for all decoder steps
    std::vector<std::vector<float>> xa_k_data(n_dec_layers);
    std::vector<std::vector<float>> xa_v_data(n_dec_layers);
    {
        fprintf(stderr, "magpie: pre-computing cross-attention K/V...\n");

        for (int l = 0; l < n_dec_layers; l++) {
            xa_k_data[l].resize(d_xa * enc_seq);
            xa_v_data[l].resize(d_xa * enc_seq);

            size_t ctx_size = ggml_tensor_overhead() * 32 + 256 * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            struct ggml_tensor * enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, enc_seq);
            ggml_set_input(enc_out);

            struct ggml_tensor * k_out = nullptr;
            struct ggml_tensor * v_out = nullptr;
            magpie_precompute_cross_attention_kv(ctx0, enc_out,
                ctx->model.decoder.layers[l].xa_kv_w,
                ctx->model.decoder.layers[l].norm_xa_mem_w,
                hp.eps, &k_out, &v_out);

            ggml_set_output(k_out);
            ggml_set_output(v_out);

            struct ggml_cgraph * gf = ggml_new_graph(ctx0);
            ggml_build_forward_expand(gf, k_out);
            ggml_build_forward_expand(gf, v_out);

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
            ggml_gallocr_reserve(allocr, gf);
            ggml_gallocr_alloc_graph(allocr, gf);

            ggml_backend_tensor_set(enc_out, ctx->state.encoder_output.data(), 0,
                                    ctx->state.encoder_output.size() * sizeof(float));
            ggml_backend_graph_compute(ctx->model.backend, gf);

            ggml_backend_tensor_get(k_out, xa_k_data[l].data(), 0, xa_k_data[l].size() * sizeof(float));
            ggml_backend_tensor_get(v_out, xa_v_data[l].data(), 0, xa_v_data[l].size() * sizeof(float));

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
        }
        fprintf(stderr, "magpie: cross-attention K/V pre-computed for %d layers\n", n_dec_layers);
    }

    // Step 4: Initialize audio sequence with BOS
    std::vector<int32_t> audio_codes;
    for (int cb = 0; cb < 8; cb++) {
        audio_codes.push_back(hp.audio_bos_id);
    }

    // Step 5: Process initial context + BOS through decoder to prime KV cache
    // For now, we'll process frame-by-frame to build up the cache
    // In a more optimized version, we could batch the context frames

    // KV cache storage: [n_layers][d_model * cache_len]
    std::vector<std::vector<float>> sa_k_cache(n_dec_layers);
    std::vector<std::vector<float>> sa_v_cache(n_dec_layers);
    int cache_len = 0;

    fprintf(stderr, "magpie: priming KV cache with context (%d frames)...\n", hp.context_frames);

    // Process context frames one by one to build cache
    for (int t = 0; t < hp.context_frames; t++) {
        // Get context embedding for this position
        std::vector<float> input_data(d_model);
        for (int d = 0; d < d_model; d++) {
            // context_data is [T * D] in row-major, need to extract column
            input_data[d] = context_data[t * d_model + d];
        }

        // Run single-step decoder
        std::vector<float> hidden_out(d_model);
        {
            size_t ctx_size = ggml_tensor_overhead() * 4096 + 512 * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, 1);
            ggml_set_input(input);

            // Create XA K/V tensors for all layers
            std::vector<struct ggml_tensor *> xa_k_tensors(n_dec_layers);
            std::vector<struct ggml_tensor *> xa_v_tensors(n_dec_layers);
            for (int l = 0; l < n_dec_layers; l++) {
                xa_k_tensors[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_xa, enc_seq);
                xa_v_tensors[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_xa, enc_seq);
                ggml_set_input(xa_k_tensors[l]);
                ggml_set_input(xa_v_tensors[l]);
            }

            // Create SA cache input tensors
            std::vector<struct ggml_tensor *> sa_k_in(n_dec_layers);
            std::vector<struct ggml_tensor *> sa_v_in(n_dec_layers);
            for (int l = 0; l < n_dec_layers; l++) {
                if (cache_len > 0) {
                    sa_k_in[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, cache_len);
                    sa_v_in[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, cache_len);
                    ggml_set_input(sa_k_in[l]);
                    ggml_set_input(sa_v_in[l]);
                } else {
                    sa_k_in[l] = nullptr;
                    sa_v_in[l] = nullptr;
                }
            }

            // Build decoder layers
            std::vector<struct ggml_tensor *> sa_k_out(n_dec_layers);
            std::vector<struct ggml_tensor *> sa_v_out(n_dec_layers);

            struct ggml_tensor * x = input;
            for (int l = 0; l < n_dec_layers; l++) {
                x = magpie_build_decoder_layer_cached(ctx0, x,
                    xa_k_tensors[l], xa_v_tensors[l],
                    sa_k_in[l], sa_v_in[l],
                    l, &ctx->model.decoder.layers[l], &hp,
                    t, ctx->model.decoder.pos_emb_w,
                    &sa_k_out[l], &sa_v_out[l]);
                if (!x) {
                    fprintf(stderr, "magpie: cached decoder layer %d failed\n", l);
                    ggml_free(ctx0);
                    return {};
                }
            }

            // Final norm
            x = magpie_build_layer_norm(ctx0, x, ctx->model.decoder.norm_out_w, hp.eps);
            ggml_set_output(x);

            // Mark cache outputs
            for (int l = 0; l < n_dec_layers; l++) {
                ggml_set_output(sa_k_out[l]);
                ggml_set_output(sa_v_out[l]);
            }

            struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);
            ggml_build_forward_expand(gf, x);
            for (int l = 0; l < n_dec_layers; l++) {
                ggml_build_forward_expand(gf, sa_k_out[l]);
                ggml_build_forward_expand(gf, sa_v_out[l]);
            }

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
            if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "magpie: cached decoder alloc failed at context frame %d\n", t);
                ggml_gallocr_free(allocr);
                ggml_free(ctx0);
                return {};
            }

            // Set inputs
            ggml_backend_tensor_set(input, input_data.data(), 0, d_model * sizeof(float));
            for (int l = 0; l < n_dec_layers; l++) {
                ggml_backend_tensor_set(xa_k_tensors[l], xa_k_data[l].data(), 0, xa_k_data[l].size() * sizeof(float));
                ggml_backend_tensor_set(xa_v_tensors[l], xa_v_data[l].data(), 0, xa_v_data[l].size() * sizeof(float));
                if (cache_len > 0) {
                    ggml_backend_tensor_set(sa_k_in[l], sa_k_cache[l].data(), 0, sa_k_cache[l].size() * sizeof(float));
                    ggml_backend_tensor_set(sa_v_in[l], sa_v_cache[l].data(), 0, sa_v_cache[l].size() * sizeof(float));
                }
            }

            ggml_backend_graph_compute(ctx->model.backend, gf);

            // Get outputs
            ggml_backend_tensor_get(x, hidden_out.data(), 0, d_model * sizeof(float));

            // Update cache
            int new_cache_len = cache_len + 1;
            for (int l = 0; l < n_dec_layers; l++) {
                sa_k_cache[l].resize(d_model * new_cache_len);
                sa_v_cache[l].resize(d_model * new_cache_len);
                ggml_backend_tensor_get(sa_k_out[l], sa_k_cache[l].data(), 0, sa_k_cache[l].size() * sizeof(float));
                ggml_backend_tensor_get(sa_v_out[l], sa_v_cache[l].data(), 0, sa_v_cache[l].size() * sizeof(float));
            }
            cache_len = new_cache_len;

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
        }

        if ((t + 1) % 20 == 0) {
            fprintf(stderr, "magpie: primed %d/%d context frames...\n", t + 1, hp.context_frames);
        }
    }

    // Process BOS frame
    {
        std::vector<float> bos_emb(d_model);
        int32_t bos_codes[8];
        for (int cb = 0; cb < 8; cb++) bos_codes[cb] = hp.audio_bos_id;
        compute_single_frame_audio_embedding(ctx, bos_codes, bos_emb);

        // Run through decoder (same as context frames)
        size_t ctx_size = ggml_tensor_overhead() * 4096 + 512 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, 1);
        ggml_set_input(input);

        std::vector<struct ggml_tensor *> xa_k_tensors(n_dec_layers);
        std::vector<struct ggml_tensor *> xa_v_tensors(n_dec_layers);
        std::vector<struct ggml_tensor *> sa_k_in(n_dec_layers);
        std::vector<struct ggml_tensor *> sa_v_in(n_dec_layers);
        std::vector<struct ggml_tensor *> sa_k_out(n_dec_layers);
        std::vector<struct ggml_tensor *> sa_v_out(n_dec_layers);

        for (int l = 0; l < n_dec_layers; l++) {
            xa_k_tensors[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_xa, enc_seq);
            xa_v_tensors[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_xa, enc_seq);
            ggml_set_input(xa_k_tensors[l]);
            ggml_set_input(xa_v_tensors[l]);

            sa_k_in[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, cache_len);
            sa_v_in[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, cache_len);
            ggml_set_input(sa_k_in[l]);
            ggml_set_input(sa_v_in[l]);
        }

        struct ggml_tensor * x = input;
        for (int l = 0; l < n_dec_layers; l++) {
            x = magpie_build_decoder_layer_cached(ctx0, x,
                xa_k_tensors[l], xa_v_tensors[l],
                sa_k_in[l], sa_v_in[l],
                l, &ctx->model.decoder.layers[l], &hp,
                cache_len, ctx->model.decoder.pos_emb_w,
                &sa_k_out[l], &sa_v_out[l]);
        }
        x = magpie_build_layer_norm(ctx0, x, ctx->model.decoder.norm_out_w, hp.eps);
        ggml_set_output(x);
        for (int l = 0; l < n_dec_layers; l++) {
            ggml_set_output(sa_k_out[l]);
            ggml_set_output(sa_v_out[l]);
        }

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);
        ggml_build_forward_expand(gf, x);
        for (int l = 0; l < n_dec_layers; l++) {
            ggml_build_forward_expand(gf, sa_k_out[l]);
            ggml_build_forward_expand(gf, sa_v_out[l]);
        }

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(input, bos_emb.data(), 0, d_model * sizeof(float));
        for (int l = 0; l < n_dec_layers; l++) {
            ggml_backend_tensor_set(xa_k_tensors[l], xa_k_data[l].data(), 0, xa_k_data[l].size() * sizeof(float));
            ggml_backend_tensor_set(xa_v_tensors[l], xa_v_data[l].data(), 0, xa_v_data[l].size() * sizeof(float));
            ggml_backend_tensor_set(sa_k_in[l], sa_k_cache[l].data(), 0, sa_k_cache[l].size() * sizeof(float));
            ggml_backend_tensor_set(sa_v_in[l], sa_v_cache[l].data(), 0, sa_v_cache[l].size() * sizeof(float));
        }

        ggml_backend_graph_compute(ctx->model.backend, gf);

        int new_cache_len = cache_len + 1;
        for (int l = 0; l < n_dec_layers; l++) {
            sa_k_cache[l].resize(d_model * new_cache_len);
            sa_v_cache[l].resize(d_model * new_cache_len);
            ggml_backend_tensor_get(sa_k_out[l], sa_k_cache[l].data(), 0, sa_k_cache[l].size() * sizeof(float));
            ggml_backend_tensor_get(sa_v_out[l], sa_v_cache[l].data(), 0, sa_v_cache[l].size() * sizeof(float));
        }
        cache_len = new_cache_len;

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }
    fprintf(stderr, "magpie: KV cache primed, cache_len=%d\n", cache_len);

    // Step 6: Autoregressive generation loop
    fprintf(stderr, "magpie: [cached] starting autoregressive decoding...\n");

    for (int step = 0; step < hp.max_dec_steps; step++) {
        // Get embedding for previous frame
        std::vector<float> prev_emb(d_model);
        int frame_idx = (int)(audio_codes.size() / 8) - 1;
        compute_single_frame_audio_embedding(ctx, &audio_codes[frame_idx * 8], prev_emb);

        // Run single-step decoder
        std::vector<float> decoder_hidden(d_model);
        {
            size_t ctx_size = ggml_tensor_overhead() * 4096 + 512 * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, 1);
            ggml_set_input(input);

            std::vector<struct ggml_tensor *> xa_k_tensors(n_dec_layers);
            std::vector<struct ggml_tensor *> xa_v_tensors(n_dec_layers);
            std::vector<struct ggml_tensor *> sa_k_in(n_dec_layers);
            std::vector<struct ggml_tensor *> sa_v_in(n_dec_layers);
            std::vector<struct ggml_tensor *> sa_k_out(n_dec_layers);
            std::vector<struct ggml_tensor *> sa_v_out(n_dec_layers);

            for (int l = 0; l < n_dec_layers; l++) {
                xa_k_tensors[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_xa, enc_seq);
                xa_v_tensors[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_xa, enc_seq);
                ggml_set_input(xa_k_tensors[l]);
                ggml_set_input(xa_v_tensors[l]);

                sa_k_in[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, cache_len);
                sa_v_in[l] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, cache_len);
                ggml_set_input(sa_k_in[l]);
                ggml_set_input(sa_v_in[l]);
            }

            struct ggml_tensor * x = input;
            for (int l = 0; l < n_dec_layers; l++) {
                x = magpie_build_decoder_layer_cached(ctx0, x,
                    xa_k_tensors[l], xa_v_tensors[l],
                    sa_k_in[l], sa_v_in[l],
                    l, &ctx->model.decoder.layers[l], &hp,
                    cache_len, ctx->model.decoder.pos_emb_w,
                    &sa_k_out[l], &sa_v_out[l]);
            }
            x = magpie_build_layer_norm(ctx0, x, ctx->model.decoder.norm_out_w, hp.eps);
            ggml_set_output(x);
            for (int l = 0; l < n_dec_layers; l++) {
                ggml_set_output(sa_k_out[l]);
                ggml_set_output(sa_v_out[l]);
            }

            struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);
            ggml_build_forward_expand(gf, x);
            for (int l = 0; l < n_dec_layers; l++) {
                ggml_build_forward_expand(gf, sa_k_out[l]);
                ggml_build_forward_expand(gf, sa_v_out[l]);
            }

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->model.backend));
            if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "magpie: cached decoder alloc failed at step %d\n", step);
                ggml_gallocr_free(allocr);
                ggml_free(ctx0);
                break;
            }

            ggml_backend_tensor_set(input, prev_emb.data(), 0, d_model * sizeof(float));
            for (int l = 0; l < n_dec_layers; l++) {
                ggml_backend_tensor_set(xa_k_tensors[l], xa_k_data[l].data(), 0, xa_k_data[l].size() * sizeof(float));
                ggml_backend_tensor_set(xa_v_tensors[l], xa_v_data[l].data(), 0, xa_v_data[l].size() * sizeof(float));
                ggml_backend_tensor_set(sa_k_in[l], sa_k_cache[l].data(), 0, sa_k_cache[l].size() * sizeof(float));
                ggml_backend_tensor_set(sa_v_in[l], sa_v_cache[l].data(), 0, sa_v_cache[l].size() * sizeof(float));
            }

            ggml_backend_graph_compute(ctx->model.backend, gf);

            ggml_backend_tensor_get(x, decoder_hidden.data(), 0, d_model * sizeof(float));

            // Update cache
            int new_cache_len = cache_len + 1;
            for (int l = 0; l < n_dec_layers; l++) {
                sa_k_cache[l].resize(d_model * new_cache_len);
                sa_v_cache[l].resize(d_model * new_cache_len);
                ggml_backend_tensor_get(sa_k_out[l], sa_k_cache[l].data(), 0, sa_k_cache[l].size() * sizeof(float));
                ggml_backend_tensor_get(sa_v_out[l], sa_v_cache[l].data(), 0, sa_v_cache[l].size() * sizeof(float));
            }
            cache_len = new_cache_len;

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
        }

        // Sample codes using local transformer
        // Forbid EOS during first min_generated_frames (4)
        const int min_generated_frames = 4;
        bool forbid_eos = (step < min_generated_frames);
        magpie_sample_result sample_result = magpie_local_transformer_sample_all(
            ctx, decoder_hidden.data(), ctx->temperature, ctx->top_k, forbid_eos);

        if (sample_result.sampled_codes.size() != 8) {
            fprintf(stderr, "magpie: local transformer failed\n");
            break;
        }

        // Debug output
        if (step < 5 || (step % 50) == 0) {
            fprintf(stderr, "magpie: step %d codes:", step);
            for (int cb = 0; cb < 8; cb++) {
                fprintf(stderr, " %d", sample_result.sampled_codes[cb]);
            }
            fprintf(stderr, "\n");
        }

        // Check for EOS using argmax_or_multinomial_any method
        bool has_eos = false;
        for (int cb = 0; cb < 8; cb++) {
            if (sample_result.sampled_codes[cb] == hp.audio_eos_id ||
                sample_result.argmax_codes[cb] == hp.audio_eos_id) {
                has_eos = true;
                break;
            }
        }
        if (has_eos) {
            fprintf(stderr, "magpie: EOS detected at step %d\n", step);
            break;
        }

        // Add frame codes
        for (int cb = 0; cb < 8; cb++) {
            audio_codes.push_back(sample_result.sampled_codes[cb]);
        }

        if ((step + 1) % 10 == 0) {
            fprintf(stderr, "magpie: [cached] generated %d frames...\n", (int)(audio_codes.size() / 8) - 1);
        }
    }

    // Return generated codes (excluding BOS frame)
    std::vector<int32_t> result;
    for (size_t i = 8; i < audio_codes.size(); i++) {
        result.push_back(audio_codes[i]);
    }

    fprintf(stderr, "magpie: [cached] synthesis complete, %d audio frames generated\n",
            (int)(result.size() / 8));

    return result;
}

// Codec functions are implemented in magpie-codec.cpp

bool magpie_is_eos(const std::vector<int32_t> & frame_codes, int32_t eos_id) {
    for (int32_t code : frame_codes) {
        if (code == eos_id) return true;
    }
    return false;
}

ggml_tensor * magpie_get_baked_context(
    ggml_context * ctx,
    magpie_embeddings * embeddings,
    int speaker_id,
    int context_frames,
    int d_model) {
    if (!ctx || !embeddings || !embeddings->baked_context_w) {
        return nullptr;
    }

    // baked_context_w is [num_speakers, context_frames * d_model] in GGML layout
    // We need to extract row for speaker_id and reshape to [d_model, context_frames]

    // Use ggml_get_rows to get the speaker embedding
    struct ggml_tensor * speaker_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(speaker_idx, "speaker_idx");
    ggml_set_input(speaker_idx);

    // Get row for this speaker: [context_frames * d_model]
    struct ggml_tensor * flat_context = ggml_get_rows(ctx, embeddings->baked_context_w, speaker_idx);

    // Reshape to [d_model, context_frames] for GGML (column-major)
    struct ggml_tensor * context = ggml_reshape_2d(ctx, flat_context, d_model, context_frames);

    return context;
}

//
// ============================================================================
// OPTIMIZED KV-CACHED SYNTHESIS (GPU-resident cache, no CPU round-trips)
// ============================================================================
//

// Initialize persistent KV cache on GPU
// Cache layout: flat 1D tensor [n_layers * max_seq * d_model]
static bool magpie_kv_cache_init_gpu(
    magpie_kv_cache & cache,
    ggml_backend_t backend,
    int n_layers,
    int d_model,
    int max_seq,
    int n_xa_heads,
    int d_xa_head,
    int enc_seq) {

    cache.max_seq = max_seq;
    cache.seq_len = 0;
    cache.enc_seq_len = enc_seq;

    // Calculate sizes
    int64_t sa_cache_size = (int64_t)n_layers * max_seq * d_model;
    int64_t xa_cache_size = (int64_t)n_layers * enc_seq * n_xa_heads * d_xa_head;

    // Allocate context for cache tensors
    size_t ctx_size = ggml_tensor_overhead() * (4 + 4) + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    cache.ctx = ggml_init(params);
    if (!cache.ctx) {
        fprintf(stderr, "magpie: failed to init KV cache context\n");
        return false;
    }

    // Create flat cache tensors (will be accessed via views)
    // Self-attention K/V: [n_layers * max_seq * d_model]
    cache.k_cache.resize(1);
    cache.v_cache.resize(1);
    cache.k_cache[0] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, sa_cache_size);
    cache.v_cache[0] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, sa_cache_size);
    ggml_set_name(cache.k_cache[0], "kv_cache_k");
    ggml_set_name(cache.v_cache[0], "kv_cache_v");

    // Cross-attention K/V: [n_layers * enc_seq * d_xa]
    cache.xa_k_cache.resize(1);
    cache.xa_v_cache.resize(1);
    cache.xa_k_cache[0] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, xa_cache_size);
    cache.xa_v_cache[0] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, xa_cache_size);
    ggml_set_name(cache.xa_k_cache[0], "xa_cache_k");
    ggml_set_name(cache.xa_v_cache[0], "xa_cache_v");

    // Allocate on backend (GPU)
    cache.buffer = ggml_backend_alloc_ctx_tensors(cache.ctx, backend);
    if (!cache.buffer) {
        fprintf(stderr, "magpie: failed to allocate KV cache on backend\n");
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
        return false;
    }

    // Clear cache
    ggml_backend_buffer_clear(cache.buffer, 0);

    fprintf(stderr, "magpie: KV cache allocated on GPU - SA: %.1f MB, XA: %.1f MB\n",
            sa_cache_size * sizeof(float) / 1024.0f / 1024.0f,
            xa_cache_size * sizeof(float) / 1024.0f / 1024.0f);

    return true;
}

static void magpie_kv_cache_free_gpu(magpie_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.xa_k_cache.clear();
    cache.xa_v_cache.clear();
}

// Build self-attention with persistent GPU cache using ggml_cpy
// This writes new K/V directly into the cache via views, avoiding CPU round-trips
static ggml_tensor * magpie_build_self_attention_gpu_cached(
    ggml_context * ctx,
    ggml_cgraph * gf,
    ggml_tensor * input,           // [d_model, 1] - single step
    ggml_tensor * qkv_weight,      // [3*d_model, d_model]
    ggml_tensor * out_weight,      // [d_model, d_model]
    ggml_tensor * kv_cache_k,      // flat cache [n_layers * max_seq * d_model]
    ggml_tensor * kv_cache_v,      // flat cache [n_layers * max_seq * d_model]
    int layer_idx,
    int cache_pos,                 // current position in cache
    int max_seq,
    int n_heads) {

    if (!input || !qkv_weight || !out_weight || !kv_cache_k || !kv_cache_v) return nullptr;

    const int64_t d_model = input->ne[0];
    const int64_t d_head = d_model / n_heads;
    const int64_t kv_len = cache_pos + 1;  // K/V sequence length after this step

    // Compute QKV for new token: [3*d_model, 1]
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_weight, input);

    // Split into Q, K_new, V_new each [d_model, 1]
    struct ggml_tensor * q = ggml_view_2d(ctx, qkv, d_model, 1, qkv->nb[1], 0);
    struct ggml_tensor * k_new = ggml_view_2d(ctx, qkv, d_model, 1, qkv->nb[1], d_model * sizeof(float));
    struct ggml_tensor * v_new = ggml_view_2d(ctx, qkv, d_model, 1, qkv->nb[1], 2 * d_model * sizeof(float));

    q = ggml_cont(ctx, q);
    k_new = ggml_cont(ctx, k_new);
    v_new = ggml_cont(ctx, v_new);

    // Calculate offset into flat cache for this layer and position
    // Layout: [layer][seq][d_model]
    size_t layer_offset = (size_t)layer_idx * max_seq * d_model * sizeof(float);
    size_t pos_offset = (size_t)cache_pos * d_model * sizeof(float);
    size_t total_offset = layer_offset + pos_offset;

    // Create views into cache at current position (for writing)
    struct ggml_tensor * k_cache_slot = ggml_view_1d(ctx, kv_cache_k, d_model, total_offset);
    struct ggml_tensor * v_cache_slot = ggml_view_1d(ctx, kv_cache_v, d_model, total_offset);

    // Copy new K/V into cache slots (this runs on GPU, no CPU round-trip!)
    struct ggml_tensor * k_cpy = ggml_cpy(ctx, ggml_reshape_1d(ctx, k_new, d_model), k_cache_slot);
    struct ggml_tensor * v_cpy = ggml_cpy(ctx, ggml_reshape_1d(ctx, v_new, d_model), v_cache_slot);

    // Add copy operations to graph
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    // Create views for all cached K/V [0, kv_len) for attention
    struct ggml_tensor * k_all = ggml_view_2d(ctx, kv_cache_k, d_model, kv_len,
                                              d_model * sizeof(float), layer_offset);
    struct ggml_tensor * v_all = ggml_view_2d(ctx, kv_cache_v, d_model, kv_len,
                                              d_model * sizeof(float), layer_offset);

    // Reshape for multi-head attention
    // Q: [d_head, n_heads, 1], K/V: [d_head, n_heads, kv_len]
    q = ggml_reshape_3d(ctx, q, d_head, n_heads, 1);
    struct ggml_tensor * k = ggml_reshape_3d(ctx, k_all, d_head, n_heads, kv_len);
    struct ggml_tensor * v = ggml_reshape_3d(ctx, v_all, d_head, n_heads, kv_len);

    // Permute to [d_head, seq, n_heads] for attention
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [d_head, 1, n_heads]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [d_head, kv_len, n_heads]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));  // [d_head, kv_len, n_heads]

    float scale = 1.0f / sqrtf((float)d_head);

    // Compute attention: scores = K.T @ Q -> [kv_len, 1, n_heads]
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, scale);

    // No causal mask needed - new token can attend to all previous tokens
    scores = ggml_soft_max(ctx, scores);

    // Apply attention to values: V @ scores -> [d_head, 1, n_heads]
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [kv_len, d_head, n_heads]
    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, v_perm, scores);

    // Reshape back to [d_model, 1]
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));  // [d_head, n_heads, 1]
    attn_out = ggml_reshape_2d(ctx, attn_out, d_model, 1);

    // Output projection
    return ggml_mul_mat(ctx, out_weight, attn_out);
}

// Build single decoder layer with GPU-resident cache
// Note: Position embedding should be added BEFORE calling this function (once per step)
static ggml_tensor * magpie_build_decoder_layer_gpu_cached(
    ggml_context * ctx,
    ggml_cgraph * gf,
    ggml_tensor * input,           // [d_model, 1] - with position embedding already added
    ggml_tensor * xa_k_cached,     // [d_xa, enc_seq]
    ggml_tensor * xa_v_cached,     // [d_xa, enc_seq]
    ggml_tensor * kv_cache_k,      // flat SA cache
    ggml_tensor * kv_cache_v,      // flat SA cache
    int layer_idx,
    int cache_pos,
    int max_seq,
    magpie_decoder_layer * layer,
    const magpie_hparams * hp) {

    struct ggml_tensor * x = input;

    // Self-attention block with GPU cache
    struct ggml_tensor * residual = x;
    x = magpie_build_layer_norm(ctx, x, layer->norm_self_w, hp->eps);
    x = magpie_build_self_attention_gpu_cached(ctx, gf, x,
        layer->sa_qkv_w, layer->sa_out_w,
        kv_cache_k, kv_cache_v,
        layer_idx, cache_pos, max_seq,
        hp->dec_sa_heads);
    if (!x) return nullptr;
    x = ggml_add(ctx, x, residual);

    // Cross-attention block (XA K/V already cached, just use them)
    residual = x;
    struct ggml_tensor * norm_q = magpie_build_layer_norm(ctx, x, layer->norm_xa_q_w, hp->eps);
    x = magpie_build_cross_attention_cached(ctx, norm_q,
        xa_k_cached, xa_v_cached,
        layer->xa_q_w, layer->xa_out_w,
        hp->dec_xa_heads, hp->dec_xa_d_head);
    if (!x) return nullptr;
    x = ggml_add(ctx, x, residual);

    // FFN block
    residual = x;
    x = magpie_build_layer_norm(ctx, x, layer->norm_ff_w, hp->eps);
    x = magpie_build_conv_ffn(ctx, x, layer->ff_proj_w, layer->ff_out_w, hp->dec_kernel);
    x = ggml_add(ctx, x, residual);

    return x;
}

// Optimized synthesis using persistent GPU-resident KV cache
std::vector<int32_t> magpie_synthesize_codes_optimized(
    magpie_context * mctx,
    const int32_t * tokens,
    int n_tokens) {

    if (!mctx || !tokens || n_tokens <= 0) {
        fprintf(stderr, "magpie_synthesize_codes_optimized: invalid args\n");
        return {};
    }

    const auto & hp = mctx->model.hparams;
    const int d_model = hp.d_model;
    const int n_dec_layers = hp.dec_layers;
    const int d_xa = hp.dec_xa_heads * hp.dec_xa_d_head;
    const int max_seq = hp.context_frames + hp.max_dec_steps + 16;  // margin for safety

    // Step 1: Encode text
    fprintf(stderr, "magpie: [optimized] encoding text (%d tokens)...\n", n_tokens);
    if (!magpie_encode_text(mctx, tokens, n_tokens)) {
        fprintf(stderr, "magpie_synthesize_codes_optimized: text encoding failed\n");
        return {};
    }
    const int enc_seq = mctx->state.enc_seq_len;
    fprintf(stderr, "magpie: text encoded, output shape [%d, %d]\n", d_model, enc_seq);

    // Step 2: Initialize GPU-resident KV cache
    magpie_kv_cache cache;
    if (!magpie_kv_cache_init_gpu(cache, mctx->model.backend,
                                   n_dec_layers, d_model, max_seq,
                                   hp.dec_xa_heads, hp.dec_xa_d_head, enc_seq)) {
        fprintf(stderr, "magpie: failed to init GPU KV cache\n");
        return {};
    }

    // Step 3: Pre-compute cross-attention K/V and store in GPU cache
    fprintf(stderr, "magpie: pre-computing cross-attention K/V...\n");
    {
        for (int l = 0; l < n_dec_layers; l++) {
            size_t ctx_size = ggml_tensor_overhead() * 32 + 256 * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            struct ggml_tensor * enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, enc_seq);
            ggml_set_input(enc_out);

            struct ggml_tensor * k_out = nullptr;
            struct ggml_tensor * v_out = nullptr;
            magpie_precompute_cross_attention_kv(ctx0, enc_out,
                mctx->model.decoder.layers[l].xa_kv_w,
                mctx->model.decoder.layers[l].norm_xa_mem_w,
                hp.eps, &k_out, &v_out);

            // Create views into XA cache for this layer
            size_t layer_offset = (size_t)l * enc_seq * d_xa * sizeof(float);
            struct ggml_tensor * xa_k_slot = ggml_view_1d(ctx0, cache.xa_k_cache[0], enc_seq * d_xa, layer_offset);
            struct ggml_tensor * xa_v_slot = ggml_view_1d(ctx0, cache.xa_v_cache[0], enc_seq * d_xa, layer_offset);

            // Copy computed K/V into cache
            struct ggml_tensor * k_cpy = ggml_cpy(ctx0, ggml_reshape_1d(ctx0, k_out, enc_seq * d_xa), xa_k_slot);
            struct ggml_tensor * v_cpy = ggml_cpy(ctx0, ggml_reshape_1d(ctx0, v_out, enc_seq * d_xa), xa_v_slot);

            struct ggml_cgraph * gf = ggml_new_graph(ctx0);
            ggml_build_forward_expand(gf, k_cpy);
            ggml_build_forward_expand(gf, v_cpy);

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
            ggml_gallocr_reserve(allocr, gf);
            ggml_gallocr_alloc_graph(allocr, gf);

            ggml_backend_tensor_set(enc_out, mctx->state.encoder_output.data(), 0,
                                    mctx->state.encoder_output.size() * sizeof(float));
            ggml_backend_graph_compute(mctx->model.backend, gf);

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
        }
    }
    fprintf(stderr, "magpie: cross-attention K/V cached on GPU for %d layers\n", n_dec_layers);

    // Step 4: Extract baked context
    std::vector<float> context_data(hp.context_frames * d_model);
    {
        size_t ctx_size = ggml_tensor_overhead() * 16 + 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * speaker_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_input(speaker_idx);

        struct ggml_tensor * flat = ggml_get_rows(ctx0, mctx->model.embeddings.baked_context_w, speaker_idx);
        ggml_set_output(flat);

        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, flat);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        int32_t sid = mctx->speaker_id;
        ggml_backend_tensor_set(speaker_idx, &sid, 0, sizeof(int32_t));
        ggml_backend_graph_compute(mctx->model.backend, gf);
        ggml_backend_tensor_get(flat, context_data.data(), 0, context_data.size() * sizeof(float));

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
    }
    fprintf(stderr, "magpie: baked context loaded for speaker %d\n", mctx->speaker_id);

    // Step 5: Prime cache with context frames
    fprintf(stderr, "magpie: priming KV cache with %d context frames...\n", hp.context_frames);
    int cache_pos = 0;

    for (int t = 0; t < hp.context_frames; t++) {
        // Get context embedding for this frame
        std::vector<float> input_data(d_model);
        for (int d = 0; d < d_model; d++) {
            input_data[d] = context_data[t * d_model + d];
        }

        // Run single-step decoder
        size_t ctx_size = ggml_tensor_overhead() * 4096 + 512 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, 1);
        ggml_set_input(input);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

        // Build decoder layers with GPU cache
        // Add position embedding ONCE before layer loop
        struct ggml_tensor * pos_slice = ggml_view_2d(ctx0, mctx->model.decoder.pos_emb_w,
            d_model, 1, mctx->model.decoder.pos_emb_w->nb[1],
            cache_pos * mctx->model.decoder.pos_emb_w->nb[1]);
        struct ggml_tensor * x = ggml_add(ctx0, input, pos_slice);

        for (int l = 0; l < n_dec_layers; l++) {
            // Get XA cache views for this layer
            size_t xa_layer_offset = (size_t)l * enc_seq * d_xa * sizeof(float);
            struct ggml_tensor * xa_k = ggml_view_2d(ctx0, cache.xa_k_cache[0], d_xa, enc_seq,
                                                     d_xa * sizeof(float), xa_layer_offset);
            struct ggml_tensor * xa_v = ggml_view_2d(ctx0, cache.xa_v_cache[0], d_xa, enc_seq,
                                                     d_xa * sizeof(float), xa_layer_offset);

            x = magpie_build_decoder_layer_gpu_cached(ctx0, gf, x,
                xa_k, xa_v,
                cache.k_cache[0], cache.v_cache[0],
                l, cache_pos, max_seq,
                &mctx->model.decoder.layers[l], &hp);
            if (!x) {
                fprintf(stderr, "magpie: decoder layer %d failed at context frame %d\n", l, t);
                ggml_free(ctx0);
                magpie_kv_cache_free_gpu(cache);
                return {};
            }
        }

        // Final norm
        x = magpie_build_layer_norm(ctx0, x, mctx->model.decoder.norm_out_w, hp.eps);
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "magpie: alloc failed at context frame %d\n", t);
            ggml_gallocr_free(allocr);
            ggml_free(ctx0);
            magpie_kv_cache_free_gpu(cache);
            return {};
        }

        ggml_backend_tensor_set(input, input_data.data(), 0, d_model * sizeof(float));
        ggml_backend_graph_compute(mctx->model.backend, gf);

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);

        cache_pos++;
        if ((t + 1) % 20 == 0) {
            fprintf(stderr, "magpie: primed %d/%d context frames...\n", t + 1, hp.context_frames);
        }
    }
    fprintf(stderr, "magpie: KV cache primed with context, cache_pos=%d\n", cache_pos);

    // Step 6: Initialize audio sequence with BOS
    std::vector<int32_t> audio_codes;
    for (int cb = 0; cb < 8; cb++) {
        audio_codes.push_back(hp.audio_bos_id);
    }

    // Process BOS frame through decoder and capture hidden state for first sample
    std::vector<float> decoder_hidden(d_model);
    {
        std::vector<float> bos_emb(d_model);
        int32_t bos_codes[8];
        for (int cb = 0; cb < 8; cb++) bos_codes[cb] = hp.audio_bos_id;
        compute_single_frame_audio_embedding(mctx, bos_codes, bos_emb);

        size_t ctx_size = ggml_tensor_overhead() * 4096 + 512 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, 1);
        ggml_set_input(input);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

        // Add position embedding ONCE before layer loop
        struct ggml_tensor * pos_slice = ggml_view_2d(ctx0, mctx->model.decoder.pos_emb_w,
            d_model, 1, mctx->model.decoder.pos_emb_w->nb[1],
            cache_pos * mctx->model.decoder.pos_emb_w->nb[1]);
        struct ggml_tensor * x = ggml_add(ctx0, input, pos_slice);

        for (int l = 0; l < n_dec_layers; l++) {
            size_t xa_layer_offset = (size_t)l * enc_seq * d_xa * sizeof(float);
            struct ggml_tensor * xa_k = ggml_view_2d(ctx0, cache.xa_k_cache[0], d_xa, enc_seq,
                                                     d_xa * sizeof(float), xa_layer_offset);
            struct ggml_tensor * xa_v = ggml_view_2d(ctx0, cache.xa_v_cache[0], d_xa, enc_seq,
                                                     d_xa * sizeof(float), xa_layer_offset);

            x = magpie_build_decoder_layer_gpu_cached(ctx0, gf, x,
                xa_k, xa_v,
                cache.k_cache[0], cache.v_cache[0],
                l, cache_pos, max_seq,
                &mctx->model.decoder.layers[l], &hp);
        }
        x = magpie_build_layer_norm(ctx0, x, mctx->model.decoder.norm_out_w, hp.eps);
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(input, bos_emb.data(), 0, d_model * sizeof(float));
        ggml_backend_graph_compute(mctx->model.backend, gf);

        // Capture BOS hidden state for first sample
        ggml_backend_tensor_get(x, decoder_hidden.data(), 0, d_model * sizeof(float));

        ggml_gallocr_free(allocr);
        ggml_free(ctx0);

        cache_pos++;
    }
    fprintf(stderr, "magpie: BOS frame processed, cache_pos=%d\n", cache_pos);

    // Step 7: Autoregressive generation loop
    // First iteration uses BOS hidden state, subsequent use computed hidden state
    fprintf(stderr, "magpie: [optimized] starting autoregressive decoding...\n");

    const int min_generated_frames = 4;

    for (int step = 0; step < hp.max_dec_steps; step++) {
        // Sample codes using current decoder hidden state
        // (For step 0, this is the BOS hidden state captured above)
        // Forbid EOS during first min_generated_frames
        bool forbid_eos = (step < min_generated_frames);
        magpie_sample_result sample_result = magpie_local_transformer_sample_all(
            mctx, decoder_hidden.data(), mctx->temperature, mctx->top_k, forbid_eos);

        if (sample_result.sampled_codes.size() != 8) {
            fprintf(stderr, "magpie: local transformer failed at step %d\n", step);
            break;
        }

        // Debug output
        if (step < 5 || (step % 50) == 0) {
            fprintf(stderr, "magpie: step %d codes:", step);
            for (int cb = 0; cb < 8; cb++) {
                fprintf(stderr, " %d", sample_result.sampled_codes[cb]);
            }
            fprintf(stderr, "\n");
        }

        // Check for EOS using argmax_or_multinomial_any method
        bool has_eos = false;
        for (int cb = 0; cb < 8; cb++) {
            if (sample_result.sampled_codes[cb] == hp.audio_eos_id ||
                sample_result.argmax_codes[cb] == hp.audio_eos_id) {
                has_eos = true;
                break;
            }
        }
        if (has_eos) {
            fprintf(stderr, "magpie: EOS detected at step %d\n", step);
            break;
        }

        // Add frame codes
        for (int cb = 0; cb < 8; cb++) {
            audio_codes.push_back(sample_result.sampled_codes[cb]);
        }

        // Check if we need more frames
        if (step + 1 >= hp.max_dec_steps) {
            break;
        }

        // Compute next hidden state using the codes we just sampled
        // Get embedding for the frame we just generated
        std::vector<float> prev_emb(d_model);
        int frame_idx = (int)(audio_codes.size() / 8) - 1;
        compute_single_frame_audio_embedding(mctx, &audio_codes[frame_idx * 8], prev_emb);

        // Run single-step decoder
        {
            size_t ctx_size = ggml_tensor_overhead() * 4096 + 512 * 1024 * 1024;
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * ctx0 = ggml_init(params);

            struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, 1);
            ggml_set_input(input);

            struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

            // Add position embedding ONCE before layer loop
            struct ggml_tensor * pos_slice = ggml_view_2d(ctx0, mctx->model.decoder.pos_emb_w,
                d_model, 1, mctx->model.decoder.pos_emb_w->nb[1],
                cache_pos * mctx->model.decoder.pos_emb_w->nb[1]);
            struct ggml_tensor * x = ggml_add(ctx0, input, pos_slice);

            for (int l = 0; l < n_dec_layers; l++) {
                size_t xa_layer_offset = (size_t)l * enc_seq * d_xa * sizeof(float);
                struct ggml_tensor * xa_k = ggml_view_2d(ctx0, cache.xa_k_cache[0], d_xa, enc_seq,
                                                         d_xa * sizeof(float), xa_layer_offset);
                struct ggml_tensor * xa_v = ggml_view_2d(ctx0, cache.xa_v_cache[0], d_xa, enc_seq,
                                                         d_xa * sizeof(float), xa_layer_offset);

                x = magpie_build_decoder_layer_gpu_cached(ctx0, gf, x,
                    xa_k, xa_v,
                    cache.k_cache[0], cache.v_cache[0],
                    l, cache_pos, max_seq,
                    &mctx->model.decoder.layers[l], &hp);
            }
            x = magpie_build_layer_norm(ctx0, x, mctx->model.decoder.norm_out_w, hp.eps);
            ggml_set_output(x);
            ggml_build_forward_expand(gf, x);

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
            if (!ggml_gallocr_reserve(allocr, gf) || !ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "magpie: alloc failed at step %d\n", step);
                ggml_gallocr_free(allocr);
                ggml_free(ctx0);
                break;
            }

            ggml_backend_tensor_set(input, prev_emb.data(), 0, d_model * sizeof(float));
            ggml_backend_graph_compute(mctx->model.backend, gf);
            ggml_backend_tensor_get(x, decoder_hidden.data(), 0, d_model * sizeof(float));

            ggml_gallocr_free(allocr);
            ggml_free(ctx0);

            cache_pos++;
        }

        if ((step + 1) % 10 == 0) {
            fprintf(stderr, "magpie: [optimized] generated %d frames...\n", (int)(audio_codes.size() / 8) - 1);
        }
    }

    // Cleanup
    magpie_kv_cache_free_gpu(cache);

    // Return generated codes (excluding BOS frame)
    std::vector<int32_t> result;
    for (size_t i = 8; i < audio_codes.size(); i++) {
        result.push_back(audio_codes[i]);
    }

    fprintf(stderr, "magpie: [optimized] synthesis complete, %d audio frames generated\n",
            (int)(result.size() / 8));

    return result;
}
