#!/usr/bin/env python3
"""
Convert Qwen3-ASR safetensors model to pre-quantized .qmodel format.

The .qmodel format is a single flat binary designed for mmap-based loading
on Android devices, eliminating the need for runtime quantization.

Usage:
    python scripts/convert-asr-qmodel.py /path/to/qwen3-asr-0.6b/ output/model.qmodel
    python scripts/convert-asr-qmodel.py /path/to/qwen3-asr-1.7b/ output/model.qmodel
"""

import sys
import struct
import json
import numpy as np
from pathlib import Path

try:
    import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy
except ImportError:
    pass

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package required. Install with: pip install safetensors", file=sys.stderr)
    sys.exit(1)

# .qmodel constants
QMODEL_MAGIC = 0x384D5141   # "AQM8"
QMODEL_VERSION = 1
HEADER_SIZE = 128            # bytes
QK8_0 = 32                  # Q8_0 block size
BLOCK_Q8_0_SIZE = 36         # 4 (float scale) + 32 (int8 qs)

# Conv stem constants
CONV_HIDDEN = 480
CONV_KERNEL = 3


def quantize_f32_to_q8_0(data_f32):
    """Quantize float32 array to Q8_0 blocks (matching C quantize_f32_to_q8_0).

    Per 32-element block: scale = max(abs(x)) / 127, qs[i] = round(x[i] / scale).
    Returns bytes: for each block, 4-byte float scale + 32 int8 values = 36 bytes.
    """
    data = data_f32.flatten().astype(np.float32)
    assert len(data) % QK8_0 == 0, f"Length {len(data)} not divisible by {QK8_0}"
    n_blocks = len(data) // QK8_0

    result = bytearray()
    for i in range(n_blocks):
        block = data[i * QK8_0:(i + 1) * QK8_0]
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax != 0 else 0.0
        if scale != 0:
            qs = np.clip(np.round(block / scale), -128, 127).astype(np.int8)
        else:
            qs = np.zeros(QK8_0, dtype=np.int8)
        result += struct.pack('<f', float(scale))
        result += qs.tobytes()

    assert len(result) == n_blocks * BLOCK_Q8_0_SIZE
    return bytes(result)


def quantize_bf16_to_q8_0(data_bf16):
    """Quantize bfloat16 array to Q8_0 blocks (matching C quantize_bf16_to_q8_0).

    First converts bf16 to f32, then applies the same Q8_0 quantization.
    data_bf16: numpy array with dtype=uint16 (raw bf16 bits).
    """
    # Convert bf16 to f32: shift left by 16 bits
    bf16_u16 = data_bf16.flatten().astype(np.uint16)
    f32_bits = bf16_u16.astype(np.uint32) << 16
    f32_data = np.frombuffer(f32_bits.tobytes(), dtype=np.float32)
    return quantize_f32_to_q8_0(f32_data)


def bf16_to_f32_array(data_bf16):
    """Convert bf16 (uint16) array to float32 numpy array."""
    bf16_u16 = data_bf16.flatten().astype(np.uint16)
    f32_bits = bf16_u16.astype(np.uint32) << 16
    return np.frombuffer(f32_bits.tobytes(), dtype=np.float32)


def get_tensor_f32(model, name):
    """Get tensor as float32 numpy array."""
    t = model.get_tensor(name)
    arr = np.array(t)
    if arr.dtype == np.float32:
        return arr
    # bfloat16 (via ml_dtypes) or float16
    if hasattr(arr.dtype, 'name') and 'bfloat16' in arr.dtype.name:
        return arr.astype(np.float32)
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    # Fallback: try viewing as uint16 for raw bf16 bits
    raw = arr.view(np.uint16)
    return bf16_to_f32_array(raw)


def get_tensor_bf16_raw(model, name):
    """Get tensor as raw uint16 (bf16 bits)."""
    t = model.get_tensor(name)
    arr = np.array(t)
    # ml_dtypes bfloat16: view the underlying bytes as uint16
    return arr.view(np.uint16)


def detect_model_variant(model):
    """Detect whether this is 0.6B or 1.7B based on encoder layer count."""
    try:
        model.get_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight")
        return "1.7B"
    except Exception:
        return "0.6B"


def get_config(variant):
    """Return model config dict based on variant."""
    if variant == "1.7B":
        return {
            'enc_d_model': 1024,
            'enc_layers': 24,
            'enc_heads': 16,
            'enc_head_dim': 64,
            'enc_ffn_dim': 4096,
            'enc_output_dim': 2048,
            'dec_hidden': 2048,
            'dec_layers': 28,
            'dec_heads': 16,
            'dec_kv_heads': 8,
            'dec_head_dim': 128,
            'dec_intermediate': 6144,
            'vocab_size': 151936,
        }
    else:  # 0.6B
        return {
            'enc_d_model': 896,
            'enc_layers': 18,
            'enc_heads': 14,
            'enc_head_dim': 64,
            'enc_ffn_dim': 3584,
            'enc_output_dim': 1024,
            'dec_hidden': 1024,
            'dec_layers': 28,
            'dec_heads': 16,
            'dec_kv_heads': 8,
            'dec_head_dim': 128,
            'dec_intermediate': 3072,
            'vocab_size': 151936,
        }


def write_header(f, cfg):
    """Write 128-byte header."""
    header = struct.pack('<II',
        QMODEL_MAGIC,
        QMODEL_VERSION,
    )
    header += struct.pack('<IIIIIIII',
        cfg['enc_d_model'],
        cfg['enc_layers'],
        cfg['enc_heads'],
        cfg['enc_head_dim'],
        cfg['enc_ffn_dim'],
        cfg['enc_output_dim'],
        cfg['dec_hidden'],
        cfg['dec_layers'],
    )
    header += struct.pack('<IIIII',
        cfg['dec_heads'],
        cfg['dec_kv_heads'],
        cfg['dec_head_dim'],
        cfg['dec_intermediate'],
        cfg['vocab_size'],
    )
    # 17 reserved uint32s to pad to 128 bytes
    # 2 + 8 + 5 = 15 uint32s so far = 60 bytes, need 128 - 60 = 68 bytes = 17 uint32s
    header += b'\x00' * (HEADER_SIZE - len(header))
    assert len(header) == HEADER_SIZE
    f.write(header)


def write_f32(f, data, label=""):
    """Write float32 data."""
    arr = np.ascontiguousarray(data.flatten(), dtype=np.float32)
    f.write(arr.tobytes())
    if label:
        print(f"  {label}: {len(arr)} f32 values ({len(arr) * 4} bytes)")


def write_q8_0(f, q8_bytes, label=""):
    """Write pre-quantized Q8_0 bytes."""
    f.write(q8_bytes)
    n_blocks = len(q8_bytes) // BLOCK_Q8_0_SIZE
    if label:
        print(f"  {label}: {n_blocks} Q8_0 blocks ({len(q8_bytes)} bytes)")


def write_bf16_raw(f, data_bf16, label=""):
    """Write raw bf16 bytes (uint16)."""
    arr = np.ascontiguousarray(data_bf16.flatten(), dtype=np.uint16)
    f.write(arr.tobytes())
    if label:
        print(f"  {label}: {len(arr)} bf16 values ({len(arr) * 2} bytes)")


ENC_PREFIX = "thinker.audio_tower."


def convert(model_dir, output_path):
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find safetensors file(s)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"Error: No safetensors files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_dir} ({len(st_files)} shard(s))...")

    # Open all shards
    models = []
    tensor_map = {}  # name -> model index
    for i, sf_path in enumerate(st_files):
        m = safe_open(str(sf_path), framework="numpy")
        models.append(m)
        for name in m.keys():
            tensor_map[name] = i

    def get_f32(name):
        idx = tensor_map[name]
        return get_tensor_f32(models[idx], name)

    def get_bf16(name):
        idx = tensor_map[name]
        return get_tensor_bf16_raw(models[idx], name)

    def get_bf16_q8(name):
        """Load bf16 tensor and quantize to Q8_0."""
        idx = tensor_map[name]
        bf16_data = get_tensor_bf16_raw(models[idx], name)
        return quantize_bf16_to_q8_0(bf16_data)

    # Detect variant
    variant = "1.7B" if f"{ENC_PREFIX}layers.18.self_attn.q_proj.weight" in tensor_map else "0.6B"
    cfg = get_config(variant)
    print(f"Detected: Qwen3-ASR-{variant}")
    print(f"  Encoder: d_model={cfg['enc_d_model']}, layers={cfg['enc_layers']}")
    print(f"  Decoder: hidden={cfg['dec_hidden']}, layers={cfg['dec_layers']}, "
          f"intermediate={cfg['dec_intermediate']}")

    enc_d_model = cfg['enc_d_model']
    enc_layers = cfg['enc_layers']
    enc_ffn_dim = cfg['enc_ffn_dim']
    enc_output_dim = cfg['enc_output_dim']
    dec_hidden = cfg['dec_hidden']
    dec_layers = cfg['dec_layers']
    dec_intermediate = cfg['dec_intermediate']
    vocab_size = cfg['vocab_size']
    dec_head_dim = cfg['dec_head_dim']
    dec_kv_heads = cfg['dec_kv_heads']
    dec_heads = cfg['dec_heads']
    conv_proj_dim = CONV_HIDDEN * 16  # 7680

    with open(output_path, 'wb') as f:
        # --- Header ---
        print("Writing header...")
        write_header(f, cfg)

        # --- Encoder conv stem ---
        print("Writing encoder conv stem...")

        # conv1: f32 [480, 1, 3, 3]
        conv1_w = get_f32(f"{ENC_PREFIX}conv2d1.weight")
        write_f32(f, conv1_w, "conv1_weight")

        conv1_b = get_f32(f"{ENC_PREFIX}conv2d1.bias")
        write_f32(f, conv1_b, "conv1_bias")

        # conv2: f32 -> q8_0 [480, 480, 3, 3] = [480, 4320]
        conv2_w = get_f32(f"{ENC_PREFIX}conv2d2.weight")
        conv2_q8 = quantize_f32_to_q8_0(conv2_w)
        write_q8_0(f, conv2_q8, "conv2_weight_q8")

        conv2_b = get_f32(f"{ENC_PREFIX}conv2d2.bias")
        write_f32(f, conv2_b, "conv2_bias")

        # conv3: f32 -> q8_0 [480, 480, 3, 3] = [480, 4320]
        conv3_w = get_f32(f"{ENC_PREFIX}conv2d3.weight")
        conv3_q8 = quantize_f32_to_q8_0(conv3_w)
        write_q8_0(f, conv3_q8, "conv3_weight_q8")

        conv3_b = get_f32(f"{ENC_PREFIX}conv2d3.bias")
        write_f32(f, conv3_b, "conv3_bias")

        # conv_out: bf16 -> q8_0 [enc_d_model, 7680]
        conv_out_q8 = get_bf16_q8(f"{ENC_PREFIX}conv_out.weight")
        write_q8_0(f, conv_out_q8, "conv_out_weight_q8")

        # --- Encoder layers ---
        print(f"Writing {enc_layers} encoder layers...")
        for i in range(enc_layers):
            lp = f"{ENC_PREFIX}layers.{i}"
            if (i + 1) % 6 == 0 or i == enc_layers - 1:
                print(f"  Layer {i+1}/{enc_layers}...")

            # Attention Q/K/V/O: bf16 -> q8_0, biases: f32
            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.q_proj.weight"))
            write_f32(f, get_f32(f"{lp}.self_attn.q_proj.bias"))

            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.k_proj.weight"))
            write_f32(f, get_f32(f"{lp}.self_attn.k_proj.bias"))

            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.v_proj.weight"))
            write_f32(f, get_f32(f"{lp}.self_attn.v_proj.bias"))

            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.out_proj.weight"))
            write_f32(f, get_f32(f"{lp}.self_attn.out_proj.bias"))

            # Pre-attention LayerNorm
            write_f32(f, get_f32(f"{lp}.self_attn_layer_norm.weight"))
            write_f32(f, get_f32(f"{lp}.self_attn_layer_norm.bias"))

            # FFN fc1/fc2: bf16 -> q8_0, biases: f32
            write_q8_0(f, get_bf16_q8(f"{lp}.fc1.weight"))
            write_f32(f, get_f32(f"{lp}.fc1.bias"))

            write_q8_0(f, get_bf16_q8(f"{lp}.fc2.weight"))
            write_f32(f, get_f32(f"{lp}.fc2.bias"))

            # Pre-FFN LayerNorm
            write_f32(f, get_f32(f"{lp}.final_layer_norm.weight"))
            write_f32(f, get_f32(f"{lp}.final_layer_norm.bias"))

        # --- Encoder post ---
        print("Writing encoder post (ln_post, proj1, proj2)...")
        write_f32(f, get_f32(f"{ENC_PREFIX}ln_post.weight"), "ln_post_weight")
        write_f32(f, get_f32(f"{ENC_PREFIX}ln_post.bias"), "ln_post_bias")

        write_q8_0(f, get_bf16_q8(f"{ENC_PREFIX}proj1.weight"), "proj1_weight_q8")
        write_f32(f, get_f32(f"{ENC_PREFIX}proj1.bias"), "proj1_bias")

        write_q8_0(f, get_bf16_q8(f"{ENC_PREFIX}proj2.weight"), "proj2_weight_q8")
        write_f32(f, get_f32(f"{ENC_PREFIX}proj2.bias"), "proj2_bias")

        # --- Decoder ---
        print("Writing decoder...")

        # tok_embeddings as bf16 raw (for embedding lookup)
        tok_emb_bf16 = get_bf16("thinker.model.embed_tokens.weight")
        write_bf16_raw(f, tok_emb_bf16, "tok_embeddings_bf16")

        # tok_embeddings as q8_0 (for argmax)
        tok_emb_q8 = quantize_bf16_to_q8_0(tok_emb_bf16)
        write_q8_0(f, tok_emb_q8, "tok_embeddings_q8")

        # Decoder layers
        print(f"Writing {dec_layers} decoder layers...")
        for i in range(dec_layers):
            if (i + 1) % 7 == 0 or i == dec_layers - 1:
                print(f"  Layer {i+1}/{dec_layers}...")

            lp = f"thinker.model.layers.{i}"

            # Attention Q/K/V/O: bf16 -> q8_0 (no biases in decoder)
            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.q_proj.weight"))
            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.k_proj.weight"))
            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.v_proj.weight"))
            write_q8_0(f, get_bf16_q8(f"{lp}.self_attn.o_proj.weight"))

            # Per-head Q/K RMSNorm
            write_f32(f, get_f32(f"{lp}.self_attn.q_norm.weight"))
            write_f32(f, get_f32(f"{lp}.self_attn.k_norm.weight"))

            # RMSNorm
            write_f32(f, get_f32(f"{lp}.input_layernorm.weight"))
            write_f32(f, get_f32(f"{lp}.post_attention_layernorm.weight"))

            # Gate+Up fused: interleave rows then quantize
            gate_bf16 = get_bf16(f"{lp}.mlp.gate_proj.weight")
            up_bf16 = get_bf16(f"{lp}.mlp.up_proj.weight")

            # gate_bf16 shape: [intermediate, hidden] as uint16
            # up_bf16 shape: [intermediate, hidden] as uint16
            gate_bf16 = gate_bf16.reshape(dec_intermediate, dec_hidden)
            up_bf16 = up_bf16.reshape(dec_intermediate, dec_hidden)

            # Interleave: fused[2*r] = gate[r], fused[2*r+1] = up[r]
            fused_bf16 = np.empty((2 * dec_intermediate, dec_hidden), dtype=np.uint16)
            fused_bf16[0::2] = gate_bf16
            fused_bf16[1::2] = up_bf16

            fused_q8 = quantize_bf16_to_q8_0(fused_bf16)
            write_q8_0(f, fused_q8)

            # down: bf16 -> q8_0
            write_q8_0(f, get_bf16_q8(f"{lp}.mlp.down_proj.weight"))

        # Final RMSNorm
        write_f32(f, get_f32("thinker.model.norm.weight"), "decoder norm")

    file_size = output_path.stat().st_size
    print(f"\nDone! Written {output_path} ({file_size:,} bytes, {file_size / 1024 / 1024:.1f} MB)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.qmodel>", file=sys.stderr)
        print(f"  model_dir: directory containing model.safetensors (and optionally shards)", file=sys.stderr)
        print(f"  output:    path for the output .qmodel file", file=sys.stderr)
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
