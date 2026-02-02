# Magpie TTS Inference Findings

## Overview

Magpie TTS is NVIDIA's text-to-speech model that generates audio from text using a decoder-only transformer architecture with discrete audio codec tokens. The model uses a two-stage approach:

1. **Stage 1 (Main Decoder)**: A transformer decoder generates coarse audio representations frame-by-frame
2. **Stage 2 (Local Transformer, optional)**: Refines the audio codes within each frame for higher quality

## Key Components

### 1. Audio Codec Model
- **Location**: Loaded from `codecmodel_path` (separate model)
- **Purpose**: Encodes/decodes audio to/from discrete tokens
- **Properties**:
  - `num_codebooks`: Number of parallel codebooks (typically 8)
  - `codebook_size`: Number of tokens per codebook (e.g., 2048)
  - `samples_per_frame`: Audio samples per codec frame

### 2. Text Tokenizer
- **Types**:
  - `IPATokenizer`: Phoneme-based tokenizer for English
  - `AggregatedTTSTokenizer`: Multi-language tokenizer supporting multiple languages
  - `CharAwareSubwordEncoder` (optional): BPE with character-level encoding
- **Special Tokens**: BOS (beginning of sequence), EOS (end of sequence)

### 3. Text Encoder (`self.encoder`)
- **Type**: `transformer_2501.Transformer`
- **Architecture**: Standard transformer encoder (non-causal self-attention)
- **Input**: Embedded text tokens
- **Output**: Contextualized text representations for decoder cross-attention

### 4. Text Embeddings
- **`self.text_embedding`**: nn.Embedding for text tokens (if not using CharAwareSubwordEncoder)
- **`self.cas_encoder`**: CharAwareSubwordEncoder (if `use_bpe_char_tokenizer=True`)
- **`self.context_text_embedding`** (optional): For text conditioning

### 5. Audio Embeddings (`self.audio_embeddings`)
- **Type**: `nn.ModuleList` of `nn.Embedding`
- **Count**: `num_audio_codebooks * frame_stacking_factor`
- **Vocabulary Size**: `num_all_tokens_per_codebook` (codebook_size + special tokens)

### 6. Decoder (`self.decoder`)
- **Type**: `transformer_2501.Transformer`
- **Architecture**: Causal transformer decoder with cross-attention
- **Input**: Embedded audio codes (previous frames) + optional context
- **Conditioning**: Cross-attention to text encoder output
- **Output**: Hidden states for predicting next audio frame

### 7. Final Projection (`self.final_proj`)
- **Type**: `nn.Linear`
- **Input**: Decoder hidden state
- **Output**: Logits for all codebooks simultaneously
- **Shape**: `(d_model) -> (num_codebooks * num_tokens * frame_stacking_factor)`

### 8. Local Transformer (optional, `self.local_transformer`)
- **Type**: `transformer_2501.Transformer`
- **Purpose**: Refine audio codes within a frame
- **Modes**:
  - `LocalTransformerType.AR`: Autoregressive (causal)
  - `LocalTransformerType.MASKGIT`: Iterative refinement with masking
- **Components**:
  - `local_transformer_in_projection`: Project to LT hidden dimension
  - `local_transformer_out_projections`: Separate projection per codebook

### 9. Context Encoder (optional, `self.context_encoder`)
- **Used in**: `decoder_ce` and `multi_encoder_context_tts` model types
- **Purpose**: Encode speaker/style context from reference audio or text

## Special Audio Tokens

```python
class SpecialAudioToken(Enum):
    AUDIO_BOS = 0        # Beginning of audio sequence
    AUDIO_EOS = 1        # End of audio sequence
    AUDIO_CONTEXT_BOS = 2  # Beginning of context audio
    AUDIO_CONTEXT_EOS = 3  # End of context audio
    MASK_TOKEN = 4       # For MaskGit training/inference
```

Token indices are: `base_codebook_size + token.value`

## Inference Flow

### High-Level Steps

1. **Prepare Context Tensors** (`prepare_context_tensors`)
   - Tokenize and embed text
   - Encode text with text encoder
   - Prepare context audio/text conditioning (if applicable)
   - Generate attention prior (if applicable)

2. **Initialize Audio Codes**
   - Start with `audio_bos_id` tokens for all codebooks
   - Shape: `(batch_size, num_codebooks, frame_stacking_factor)`

3. **Autoregressive Generation Loop** (main loop in `infer_batch`)
   ```
   for idx in range(max_decoder_steps // frame_stacking_factor):
       # Embed current audio codes
       audio_embedded = embed_audio_tokens(audio_codes_input)

       # Decode with cross-attention to text
       logits, attn_probs, dec_out = forward(
           dec_input_embedded=audio_embedded,
           cond=text_encoder_out,
           cond_mask=text_mask,
           attn_prior=attention_prior  # optional
       )

       # Sample next audio codes
       if use_local_transformer:
           next_codes = local_transformer_sample_*(dec_out[:, -1, :])
       else:
           next_codes = sample_codes_from_logits(logits[:, -1, :])

       # Append and check for EOS
       audio_codes_input = cat([audio_codes_input, next_codes], dim=-1)
       if eos_detected: break
   ```

4. **Convert Codes to Audio** (`codes_to_audio`)
   - Use codec model's decoder to generate waveform
   - Return audio tensor and lengths

### Sampling Methods

1. **Direct Sampling** (`sample_codes_from_logits`)
   - Top-k filtering + temperature sampling
   - All codebooks sampled in parallel

2. **Autoregressive Local Transformer** (`local_transformer_sample_autoregressive`)
   - Codebooks sampled sequentially within each frame
   - Uses KV-cache for efficiency

3. **MaskGit Local Transformer** (`local_transformer_sample_maskgit`)
   - Start with all positions masked
   - Iteratively unmask high-confidence positions
   - Cosine schedule for number of tokens to unmask

### Classifier-Free Guidance (CFG)

When `use_cfg=True`:
- Run forward pass twice: with and without conditioning
- Interpolate logits: `cfg_scale * cond_logits + (1 - cfg_scale) * uncond_logits`

### Attention Prior

Used to guide cross-attention for better alignment:
- Constructed based on previous attention patterns
- Helps prevent skipping/repeating text
- Can be applied to specific decoder layers

## Model Types

1. **`decoder_context_tts`**: Context audio goes to decoder input
2. **`decoder_ce`**: Context encoder processes context, feeds to decoder
3. **`multi_encoder_context_tts`**: Separate encoders for text and context

## Frame Stacking

- `frame_stacking_factor`: Number of frames processed together
- Reduces sequence length by factor, improves efficiency
- Requires undoing stacking in `logits_to_audio_codes`

## Key Dataclasses

### `ModelInferenceParameters`
```python
@dataclass
class ModelInferenceParameters:
    max_decoder_steps: int = 500
    temperature: float = 0.7
    topk: int = 80
    cfg_scale: float = 2.5
    apply_attention_prior: bool = True
    attention_prior_epsilon: float = 0.1
    attention_prior_lookahead_window: int = 5
    eos_detection_method: str = "argmax_or_multinomial_any"
    min_generated_frames: int = 4
```

### `InferBatchOutput`
```python
@dataclass
class InferBatchOutput:
    predicted_audio: torch.Tensor      # Generated waveforms
    predicted_audio_lens: torch.Tensor # Lengths in samples
    predicted_codes: torch.Tensor      # Generated codec tokens
    predicted_codes_lens: torch.Tensor # Lengths in frames
    rtf_metrics: Dict[str, Any]        # Real-time factor metrics
    cross_attention_maps: Optional[List[Any]] = None
```

## EOS Detection Methods

```python
class EOSDetectionMethod(PrettyStrEnum):
    ARGMAX_ANY = "argmax_any"                    # Any codebook predicts EOS via argmax
    ARGMAX_OR_MULTINOMIAL_ANY = "argmax_or_multinomial_any"  # Either sampling method
    ARGMAX_ALL = "argmax_all"                    # All codebooks predict EOS
    ARGMAX_ZERO_CB = "argmax_zero_cb"            # First codebook predicts EOS
```

## Longform Inference

For long text (>40 words by default):
- Text is chunked by sentences
- Audio is generated chunk-by-chunk with state carried over
- Uses `LongformChunkState` and `LongformDecoderState`
- Attention prior helps maintain coherence across chunks

## Files to Port

For GGML implementation, key files to understand:
1. `NeMo/nemo/collections/tts/models/magpietts.py` - Main model
2. `NeMo/nemo/collections/tts/modules/transformer_2501.py` - Transformer layers
3. `NeMo/nemo/collections/tts/modules/magpietts_modules.py` - Helper modules
4. Audio codec model (separate port needed)

## Tensor Shapes Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| text | (B, T_text) | Input text token IDs |
| text_embedded | (B, T_text, E) | Embedded text |
| text_encoder_out | (B, T_text, E) | Encoded text (conditioning) |
| audio_codes_input | (B, C, T_audio) | Current audio codes |
| audio_embedded | (B, T_audio/fsf, E) | Embedded audio (after frame stacking) |
| decoder_out | (B, T_audio/fsf, E) | Decoder hidden states |
| all_code_logits | (B, T_audio/fsf, C*V*fsf) | Logits for all codebooks |
| predicted_codes | (B, C, T_audio) | Final predicted codes |
| predicted_audio | (B, T_samples) | Generated waveform |

Where:
- B = batch size
- T_text = text sequence length
- T_audio = audio sequence length in frames
- C = num_audio_codebooks
- E = embedding dimension (d_model)
- V = vocab size per codebook (num_all_tokens_per_codebook)
- fsf = frame_stacking_factor
