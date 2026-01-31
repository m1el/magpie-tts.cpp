"""
Magpie TTS Model Inspection Script

This script loads the Magpie TTS model and instruments it to capture
the forward pass of all modules for analysis. Useful for understanding
the model architecture and porting to GGML.

Usage:
    cd magpie.cpp && uv run inspect_inference.py
"""

from collections import OrderedDict
import inspect
import struct
import os
import torch

# ==============================================================================
# Instrumentation utilities
# ==============================================================================

def mk_hook(module_name):
    """Create a forward hook that logs module info and tensor shapes."""
    def hook(module, args, kwargs, output):
        if kwargs:
            args = (*args, kwargs)
        ty = get_ty((args, output))
        append_name(module_name, module, ty)
    return hook


def hook_patch_object_method(obj, name, prefix, force=False):
    """Patch an object method to add instrumentation."""
    if not hasattr(obj, name):
        assert not force, f'Object {obj} has no method {name}'
        return
    orig_method = getattr(obj, name)
    hook_fn = mk_hook(prefix + name)
    def wrapped(*args, **kwargs):
        rv = orig_method(*args, **kwargs)
        hook_fn(orig_method, args, kwargs, rv)
        return rv
    setattr(obj, name, wrapped)


visited = set()
def instrument_everything(model, prefix=''):
    """Recursively instrument all modules in a model."""
    append_name(prefix, model)
    if prefix != '':
        prefix = prefix + '.'

    if model in visited:
        return
    visited.add(model)
    for name, module in model.named_modules():
        if module in visited:
            continue
        visited.add(module)
        if name == '':
            continue
        module.register_forward_hook(mk_hook(prefix + name), with_kwargs=True)
        instrument_everything(module, prefix=prefix + name)


all_names = OrderedDict()
def append_name(name, mod, inout=None):
    """Record module information in a hierarchical dictionary."""
    chunks = name.split('.')
    cur = all_names
    for c in chunks:
        if c not in cur:
            cur[c] = {}
        cur = cur[c]
    cur['##cls'] = mod.__class__.__name__
    fn = mod.forward if hasattr(mod, 'forward') else mod
    try:
        cur['##sig'] = inspect.signature(fn)
    except (ValueError, TypeError):
        cur['##sig'] = ''
    if inout is not None:
        cur['##inout'] = inout


def pprint_names_flat(d, prefix='', file=None):
    """Print module names in flat format."""
    if prefix != '':
        prefix = prefix + '.'
    for k, v in d.items():
        if k.startswith('##'):
            continue
        if k != '':
            inout = v.get('##inout', '')
            if inout != '':
                i, o = inout
                inout = f' | {i} -> {o}'
            sig = v.get('##sig', '')
            if sig != '':
                inout = str(sig) + inout
            line = f'{prefix}{k}: {v.get("##cls", "unknown")}{inout}'
            print(line, file=file)
        pprint_names_flat(v, prefix=prefix + k, file=file)


def pprint_names_tree(d, indent=0, file=None):
    """Print module names in tree format."""
    for k, v in d.items():
        if k.startswith('##'):
            continue
        if k != '':
            inout = v.get('##inout', '')
            if inout != '':
                i, o = inout
                inout = f' | {i} -> {o}'
            sig = v.get('##sig', '')
            if sig != '':
                inout = str(sig) + inout
            line = '  ' * indent + f'{k}: {v.get("##cls", "unknown")}{inout}'
            print(line, file=file)
        pprint_names_tree(v, indent + 1, file=file)


class TensorShape:
    """Helper class to pretty print tensor shapes."""
    def __init__(self, tensor):
        self.shape = list(tensor.shape)
        self.dtype = str(tensor.dtype).replace('torch.', '')
    def __str__(self):
        return f'Tensor[{self.dtype}]{self.shape}'
    def __repr__(self):
        return self.__str__()


def get_ty(obj):
    """Recursively get type information for nested structures."""
    if isinstance(obj, tuple):
        return tuple(get_ty(o) for o in obj)
    elif isinstance(obj, list):
        return [get_ty(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: get_ty(v) for k, v in obj.items()}
    elif torch.is_tensor(obj):
        return TensorShape(obj)
    else:
        return type(obj).__name__


output_files = {}

def dump_append_data(tensor, filename):
    """Append tensor data to a binary file for offline analysis."""
    if filename not in output_files:
        print(f'Creating file {filename} tensor shape {TensorShape(tensor)}')
        shape = list(tensor.shape)
        shape.reverse()
        while len(shape) < 4:
            shape.append(1)
        struct.pack_into('4q', b := bytearray(32), 0, *shape)
        file = open(filename, "wb")
        assert len(b) == 32
        file.write(b)
        output_files[filename] = (file, list(tensor.shape))
    file, shape = output_files[filename]

    assert list(tensor.shape) == shape, \
        f"Shape mismatch for {filename}: expected {shape}, got {list(tensor.shape)}"
    file.write(tensor.detach().cpu().numpy().tobytes())


# ==============================================================================
# Model loading and inference
# ==============================================================================

def load_magpie_model(
    model_name: str = "nvidia/magpie_tts_multilingual_357m",
    device: str = "cuda",
):
    """
    Load Magpie TTS model from pretrained or local path.

    Args:
        model_name: Model identifier (HuggingFace path or local .nemo file)
        device: Device to load model on

    Returns:
        Loaded MagpieTTSModel instance
    """
    from nemo.collections.tts.models import MagpieTTSModel

    print(f"Loading Magpie TTS model: {model_name}")

    if model_name.startswith("nvidia/"):
        # Load from HuggingFace/NGC
        model = MagpieTTSModel.from_pretrained(model_name)
    elif model_name.endswith(".nemo"):
        # Load from local .nemo file
        model = MagpieTTSModel.restore_from(model_name)
    else:
        raise ValueError(f"Unknown model format: {model_name}")

    model.use_kv_cache_for_inference = True
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"  - Sample rate: {model.sample_rate}")
    print(f"  - Output sample rate: {model.output_sample_rate}")
    print(f"  - Num codebooks: {model.num_audio_codebooks}")
    print(f"  - Codebook size: {model.codebook_size}")
    print(f"  - Frame stacking factor: {model.frame_stacking_factor}")
    print(f"  - Model type: {model.model_type}")

    return model


def create_test_batch(model, text: str, context_text: str = None):
    """
    Create a minimal test batch for inference.

    Args:
        model: Loaded MagpieTTSModel
        text: Text to synthesize
        context_text: Optional speaker/style context text

    Returns:
        Batch dictionary ready for inference
    """
    device = next(model.parameters()).device

    # Tokenize text
    tokenizer = model.tokenizer

    # Get text tokens
    if hasattr(tokenizer, 'tokenizers'):
        # AggregatedTTSTokenizer
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        # Single tokenizer
        text_tokens = tokenizer(text)

    # Add BOS and EOS
    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    text_tensor = torch.tensor([text_tokens], dtype=torch.long, device=device)
    text_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device=device)

    batch = {
        'text': text_tensor,
        'text_lens': text_lens,
        'sample_rate': model.output_sample_rate,
        'context_sample_rate': model.output_sample_rate,
    }

    # Handle context if model requires it
    if model.model_type in ['decoder_context_tts', 'decoder_ce']:
        # Create dummy context (silent audio or text context)
        # For now, use minimal context
        context_codes = torch.full(
            (1, model.num_audio_codebooks, 4),  # 4 frames minimum
            fill_value=model.context_audio_bos_id,
            dtype=torch.long,
            device=device
        )
        context_codes[:, :, -1] = model.context_audio_eos_id
        context_codes_lens = torch.tensor([4], dtype=torch.long, device=device)

        batch['context_audio_codes'] = context_codes
        batch['context_audio_codes_lens'] = context_codes_lens

        if context_text is not None:
            # Tokenize context text
            if hasattr(tokenizer, 'tokenizers') and model.text_conditioning_tokenizer_name in tokenizer.tokenizers:
                ctx_tokenizer = tokenizer.tokenizers[model.text_conditioning_tokenizer_name]
                context_text_tokens = ctx_tokenizer.encode(context_text)
            else:
                context_text_tokens = [0]  # Dummy

            batch['context_text'] = torch.tensor([context_text_tokens], dtype=torch.long, device=device)
            batch['context_text_lens'] = torch.tensor([len(context_text_tokens)], dtype=torch.long, device=device)

    return batch


def run_inference(model, batch, instrument: bool = True):
    """
    Run inference on a batch and optionally instrument the model.

    Args:
        model: Loaded MagpieTTSModel
        batch: Batch dictionary from create_test_batch
        instrument: Whether to instrument the model for analysis

    Returns:
        InferBatchOutput from the model
    """
    if instrument:
        global visited, all_names
        visited = set()
        all_names = OrderedDict()
        instrument_everything(model)

    with torch.no_grad():
        output = model.infer_batch(
            batch,
            use_cfg=False,
            return_cross_attn_probs=True,
        )

    return output


def print_model_summary(model):
    """Print a summary of the model architecture."""
    print("\n" + "=" * 80)
    print("MAGPIE TTS MODEL SUMMARY")
    print("=" * 80)

    # Print main components
    print("\n### Main Components ###")
    for name, module in model.named_children():
        if name.startswith('_'):
            continue
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({param_count:,} params)")

    # Print encoder config
    print("\n### Text Encoder Config ###")
    enc = model.encoder
    print(f"  Layers: {enc.n_layers}")
    if hasattr(enc.layers[0], 'self_attention'):
        sa = enc.layers[0].self_attention
        print(f"  Self-Attention Heads: {sa.n_heads}")
        print(f"  d_model: {sa.d_model}")
        print(f"  d_head: {sa.d_head}")

    # Print decoder config
    print("\n### Decoder Config ###")
    dec = model.decoder
    print(f"  Layers: {dec.n_layers}")
    if hasattr(dec.layers[0], 'self_attention'):
        sa = dec.layers[0].self_attention
        print(f"  Self-Attention Heads: {sa.n_heads}")
        print(f"  d_model: {sa.d_model}")
        print(f"  d_head: {sa.d_head}")
        print(f"  is_causal: {sa.is_causal}")
    if hasattr(dec.layers[0], 'cross_attention'):
        xa = dec.layers[0].cross_attention
        print(f"  Cross-Attention Heads: {xa.n_heads}")

    # Print local transformer config if present
    if hasattr(model, 'local_transformer') and model.local_transformer_type.value != 'none':
        print("\n### Local Transformer Config ###")
        lt = model.local_transformer
        print(f"  Type: {model.local_transformer_type.value}")
        print(f"  Layers: {lt.n_layers}")

    # Print audio embedding info
    print("\n### Audio Embeddings ###")
    print(f"  Num embeddings: {len(model.audio_embeddings)}")
    print(f"  Vocab size per codebook: {model.num_all_tokens_per_codebook}")
    if model.audio_embeddings:
        print(f"  Embedding dim: {model.audio_embeddings[0].embedding_dim}")

    # Print special tokens
    print("\n### Special Tokens ###")
    print(f"  BOS ID (text): {model.bos_id}")
    print(f"  EOS ID (text): {model.eos_id}")
    print(f"  Audio BOS ID: {model.audio_bos_id}")
    print(f"  Audio EOS ID: {model.audio_eos_id}")
    print(f"  Context Audio BOS ID: {model.context_audio_bos_id}")
    print(f"  Context Audio EOS ID: {model.context_audio_eos_id}")
    print(f"  Mask Token ID: {model.mask_token_id}")

    print("\n" + "=" * 80)


def main():
    """Main entry point for model inspection."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Magpie TTS model")
    parser.add_argument(
        "--model",
        type=str,
        default="../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo",
        help="Model name or path (default: ../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Magpie. a text to speech system.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inspect_output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip inference, just print model summary"
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio to file"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU (will be slow)")

    # Load model
    try:
        model = load_magpie_model(args.model, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTo use this script, you need the NeMo package with Magpie TTS support.")
        print("The model will be downloaded from HuggingFace/NGC on first use.")
        return

    # Print model summary
    print_model_summary(model)

    if args.no_inference:
        print("\nSkipping inference (--no-inference flag set)")
        return

    # Create test batch
    print(f"\nCreating test batch with text: '{args.text}'")
    batch = create_test_batch(model, args.text)

    # Run inference with instrumentation
    print("\nRunning inference with instrumentation...")
    output = run_inference(model, batch, instrument=True)

    # Print instrumented module information
    print("\n" + "=" * 80)
    print("INSTRUMENTED MODULE TRACE")
    print("=" * 80)

    # Save flat output to file
    flat_output_path = os.path.join(args.output_dir, "modules_flat.txt")
    with open(flat_output_path, "w") as f:
        pprint_names_flat(all_names, file=f)
    print(f"\nFlat module list saved to: {flat_output_path}")

    # Also print to console (truncated)
    print("\n### Module Trace (flat, first 50 lines) ###")
    lines = []
    import io
    buf = io.StringIO()
    pprint_names_flat(all_names, file=buf)
    lines = buf.getvalue().split('\n')
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"... ({len(lines) - 50} more lines, see {flat_output_path})")

    # Print inference results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    print(f"  Predicted audio shape: {output.predicted_audio.shape}")
    print(f"  Predicted audio length: {output.predicted_audio_lens[0].item()} samples")
    print(f"  Predicted codes shape: {output.predicted_codes.shape}")
    print(f"  Predicted codes length: {output.predicted_codes_lens[0].item()} frames")
    print(f"  RTF metrics: {output.rtf_metrics}")

    if args.save_audio:
        import soundfile as sf
        audio_path = os.path.join(args.output_dir, "generated_audio.wav")
        audio = output.predicted_audio[0].float().cpu().numpy()
        audio = audio[:output.predicted_audio_lens[0].item()]
        sf.write(audio_path, audio, model.output_sample_rate)
        print(f"\nGenerated audio saved to: {audio_path}")

    print("\nInspection complete!")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up any open files
        for _filename, (file, _shape) in output_files.items():
            file.close()
        output_files.clear()
