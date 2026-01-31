# Makefile for ggml-based Magpie TTS implementation

GGML_DIR = ggml
GGML_BUILD = $(GGML_DIR)/build

CXX = g++
CXXFLAGS = -g -std=c++17 -Wall -Wextra -O2
CXXFLAGS += -I $(GGML_DIR)/include
CXXFLAGS += -I src

# Check if CUDA backend is available
CUDA_LIB = $(GGML_BUILD)/src/ggml-cuda/libggml-cuda.so
CUDA_AVAILABLE = $(shell test -f $(CUDA_LIB) && echo 1 || echo 0)

METAL_LIB = $(GGML_BUILD)/src/ggml-metal/libggml-metal.dylib
METAL_AVAILABLE = $(shell test -f $(METAL_LIB) && echo 1 || echo 0)

LDFLAGS = -L $(GGML_BUILD)/src
LDFLAGS += -lggml -lggml-base -lggml-cpu
LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/src
LDFLAGS += -lm -lpthread

# Add CUDA support if available
ifeq ($(CUDA_AVAILABLE),1)
    CXXFLAGS += -DGGML_USE_CUDA
    LDFLAGS += -L $(GGML_BUILD)/src/ggml-cuda -lggml-cuda
    LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/src/ggml-cuda
    LDFLAGS += -L /usr/local/cuda/lib64 -lcudart -lcublas
    LDFLAGS += -Wl,-rpath,/usr/local/cuda/lib64
endif

# Add Metal support if available
ifeq ($(METAL_AVAILABLE),1)
    CXXFLAGS += -DGGML_USE_METAL
    LDFLAGS += -L $(GGML_BUILD)/src/ggml-metal -lggml-metal
    LDFLAGS += -Wl,-rpath,$(GGML_BUILD)/src/ggml-metal
    LDFLAGS += -framework Metal -framework Foundation
endif

# Source files
SRCS = src/magpie.cpp src/magpie-codec.cpp

.PHONY: all clean check-header

all: magpie-tts

# Quick header syntax check (compile only, no link)
check-header:
	$(CXX) $(CXXFLAGS) -fsyntax-only src/magpie.h

# Main binary
magpie-tts: examples/main.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test loading weights
test_load: tests/test_load.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test text embedding
test_text_embedding: tests/test_text_embedding.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test audio embedding
test_audio_embedding: tests/test_audio_embedding.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test RMS norm
test_rms_norm: tests/test_rms_norm.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test Layer norm (correct implementation)
test_layer_norm: tests/test_layer_norm.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test encoder layer
test_encoder_layer: tests/test_encoder_layer.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test self-attention
test_self_attention: tests/test_self_attention.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test conv FFN
test_conv_ffn: tests/test_conv_ffn.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test full encoder (6 layers)
test_full_encoder: tests/test_full_encoder.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test decoder layer
test_decoder_layer: tests/test_decoder_layer.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test full encoder v2
test_full_encoder_v2: tests/test_full_encoder_v2.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test full decoder (12 layers)
test_full_decoder: tests/test_full_decoder.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test final projection
test_final_proj: tests/test_final_proj.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test local transformer
test_local_transformer: tests/test_local_transformer.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test codec loading
test_codec_load: tests/test_codec_load.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test codec FSQ dequantization
test_codec_fsq: tests/test_codec_fsq.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test codec decoder
test_codec_decode: tests/test_codec_decode.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test end-to-end inference
test_e2e_inference: tests/test_e2e_inference.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Test end-to-end inference with KV cache (optimized)
test_e2e_cached: tests/test_e2e_cached.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f magpie-tts test_load test_text_embedding test_audio_embedding test_rms_norm test_encoder_layer test_self_attention test_conv_ffn test_full_encoder
