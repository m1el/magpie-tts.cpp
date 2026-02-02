# NeMo Nano Codec Architecture

## Overview

The NeMo nano-codec is a neural audio codec based on HiFi-GAN architecture with Finite Scalar Quantization (FSQ).
It converts audio waveforms to discrete codes (encoder) and back (decoder) at approximately 21.5 frames per second.

**Key Parameters:**
- Sample rate: 22050 Hz
- Hop length: 1024 samples (~46.4 ms per frame, ~21.5 fps)
- Num codebooks: 8
- Codebook size: 2016 (per codebook)
- Latent dimension: 32 (8 codebooks × 4 dims per codebook)
- Total upsample factor: 8×8×4×2×2 = 1024 (matches hop length)

## Architecture Diagram

```
Audio Codes [B, 8, T]
      │
      ▼
┌─────────────────────────────┐
│  FSQ Dequantization         │  indices → continuous values
│  8 × FiniteScalarQuantizer  │  [B, 8, T] → [B, 32, T]
│  levels=[8,7,6,6] per cb    │
└──────────────┬──────────────┘
               │
               ▼
         [B, 32, T]
               │
               ▼
┌─────────────────────────────┐
│  Pre-Conv                   │  CausalConv1d(32→864, kernel=7)
└──────────────┬──────────────┘
               │
               ▼
         [B, 864, T]
               │
      ┌────────┴────────┐
      ▼                 │
┌───────────────┐       │
│ HalfSnake Act │       │  (Snake on first half, LeakyReLU on second)
└───────┬───────┘       │
        │               │
        ▼               │
┌───────────────┐       │
│ UpConvT(×8)   │       │  CausalConvTranspose1d(864→432, kernel=16, stride=8)
└───────┬───────┘       │
        │               │
        ▼               │
  [B, 432, T×8]         │
        │               │
        ▼               │
┌───────────────┐       │
│ ResLayer 0    │       │  3 HiFiGANResBlocks (kernels=[3,7,11])
└───────┬───────┘       │
        │               │
        ├───────────────┘  (repeat 4 more times with halving channels)
        ▼
... (4 more upsample stages) ...
        │
        ▼
  [B, 27, T×1024]
        │
        ▼
┌───────────────┐
│ HalfSnake Act │  Post-activation
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Post-Conv     │  CausalConv1d(27→1, kernel=3)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Tanh          │  Output activation
└───────┬───────┘
        │
        ▼
  Audio [B, T×1024]
```

## Component Details

### 1. FSQ Dequantization

Each of the 8 codebooks uses Finite Scalar Quantization with levels `[8, 7, 6, 6]`:
- 4 dimensions per codebook
- Total codebook size: 8×7×6×6 = 2016
- Dequantization formula:
  ```
  # For index i and levels L = [8, 7, 6, 6]:
  dim_base_index = [1, 8, 56, 336]  # cumulative product

  # Convert single index to per-dimension nonnegative indices
  nonneg[d] = (index // dim_base_index[d]) % L[d]

  # Convert to centered codes [-1, 1]
  code[d] = (nonneg[d] - L[d]//2) / (L[d]//2)
  ```

### 2. CausalHiFiGANDecoder

**Pre-Conv Layer:**
- `CausalConv1dNorm(in=32, out=864, kernel=7)`
- Left-pad by 6 for causality

**Upsample Stages (5 total):**

| Stage | In Ch | Out Ch | Kernel | Stride | Upsample |
|-------|-------|--------|--------|--------|----------|
| 0     | 864   | 432    | 16     | 8      | ×8       |
| 1     | 432   | 216    | 16     | 8      | ×8       |
| 2     | 216   | 108    | 8      | 4      | ×4       |
| 3     | 108   | 54     | 4      | 2      | ×2       |
| 4     | 54    | 27     | 4      | 2      | ×2       |

Each stage consists of:
1. **HalfSnake activation** on previous output
2. **CausalConvTranspose1d** upsample
3. **HiFiGANResLayer** (3 residual blocks averaged)

**Post Processing:**
- HalfSnake activation
- `CausalConv1dNorm(in=27, out=1, kernel=3)`
- Tanh output activation

### 3. HalfSnake Activation

Applies Snake to the first half of channels and LeakyReLU to the second half:

```python
def half_snake(x, alpha):
    # x: [B, C, T], alpha: [C//2]
    first_half = x[:, :C//2, :]
    second_half = x[:, C//2:, :]

    # Snake activation: x + (1/alpha) * sin²(alpha * x)
    first_half = first_half + (1/alpha) * torch.sin(alpha * first_half)**2

    # LeakyReLU on second half
    second_half = F.leaky_relu(second_half, 0.1)

    return torch.cat([first_half, second_half], dim=1)
```

### 4. HiFiGANResLayer

Each ResLayer contains 3 HiFiGANResBlocks with different kernel sizes `[3, 7, 11]`.
The outputs are averaged:

```
output = (resblock_k3(x) + resblock_k7(x) + resblock_k11(x)) / 3
```

### 5. HiFiGANResBlock

Each ResBlock contains 3 inner ResidualBlocks with dilations `[1, 3, 5]`:

```
for dilation in [1, 3, 5]:
    x = residual_block(x, dilation)
```

### 6. ResidualBlock

```python
def residual_block(x, dilation):
    # x: [B, C, T]

    # Input path
    h = half_snake(x)
    h = causal_conv1d(h, dilation=dilation)  # dilated conv

    # Skip path
    h = half_snake(h)
    h = causal_conv1d(h, dilation=1)  # non-dilated conv

    # Residual connection
    return x + h
```

## Tensor Flow Example (5 input frames)

```
Input codes:          [1, 8, 5]
After FSQ dequant:    [1, 32, 5]
After pre_conv:       [1, 864, 5]
After upsample[0]:    [1, 432, 40]    (×8)
After upsample[1]:    [1, 216, 320]   (×8)
After upsample[2]:    [1, 108, 1280]  (×4)
After upsample[3]:    [1, 54, 2560]   (×2)
After upsample[4]:    [1, 27, 5120]   (×2)
After post_conv:      [1, 1, 5120]
Output audio:         [1, 5120]       (5 × 1024 = 5120 samples)
```

## GGUF Tensor Names

### Pre/Post Convolutions
```
audio_decoder.pre_conv.conv.weight: [864, 32, 7]
audio_decoder.pre_conv.conv.bias: [864]
audio_decoder.post_conv.conv.weight: [1, 27, 3]
audio_decoder.post_conv.conv.bias: [1]
audio_decoder.post_activation.activation.snake_act.alpha: [1, 13, 1]  # half of 27
```

### Upsample Layers
```
audio_decoder.up_sample_conv_layers.{i}.conv.weight: [out_ch, 1, kernel]  # groups=out_ch
audio_decoder.up_sample_conv_layers.{i}.conv.bias: [out_ch]
audio_decoder.activations.{i}.activation.snake_act.alpha: [1, in_ch//2, 1]
```

### Residual Layers
For each res_layer `i` (0-4), res_block `j` (0-2), inner block `k` (0-2):
```
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.input_activation.activation.snake_act.alpha
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.input_conv.conv.weight
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.input_conv.conv.bias
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.skip_activation.activation.snake_act.alpha
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.skip_conv.conv.weight
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{k}.skip_conv.conv.bias
```

### FSQ Parameters
```
vector_quantizer.fsqs.{i}.dim_base_index: [1, 4, 1]  # [1, 8, 56, 336]
vector_quantizer.fsqs.{i}.num_levels: [1, 4, 1]      # [8, 7, 6, 6]
```

## Total Tensor Count

- Pre-conv: 2 tensors
- Post-conv: 3 tensors (weight, bias, post_activation alpha)
- Upsample layers: 5 × 3 = 15 tensors (weight, bias, activation alpha)
- Residual layers: 5 × 3 × 3 × 6 = 270 tensors
- FSQ parameters: 8 × 2 = 16 tensors

**Total: 306 tensors**

## Implementation Notes

1. **Weight Normalization**: Convolution weights in the GGUF are already combined from weight_norm (g×v/||v||).

2. **Causal Convolution**: Left-pad input by `kernel_size - 1` (or `(kernel_size - 1) × dilation` for dilated).

3. **ConvTranspose Trimming**: After transposed conv, trim `padding_right` samples from right.

4. **HalfSnake Channels**: For a layer with C channels:
   - First C/2 channels: Snake activation with alpha of shape [1, C/2, 1]
   - Last C/2 channels: LeakyReLU (slope=0.01, PyTorch default)

5. **Residual Averaging**: HiFiGANResLayer averages 3 parallel residual blocks (kernels 3, 7, 11).
