#!/usr/bin/env python3
"""
convert-asr-gguf.py - Convert Qwen3-ASR safetensors to GGUF format

Quantization strategy:
  - Encoder linear weights: Q8_0 (block size 32)
  - Decoder linear weights: Q4_K (block size 256)
  - Conv weights, norms, biases: F32
  - Token embeddings: F16 (for lookup) + Q4_K (for argmax)

Usage:
  pip install gguf safetensors numpy ml-dtypes
  python convert-asr-gguf.py <model_dir> <output.gguf>

Example:
  python convert-asr-gguf.py ~/qwen-asr/qwen3-asr-0.6b/ /tmp/model.gguf
"""

import sys
import os
import json
import struct
import numpy as np
import ml_dtypes  # noqa: F401 — registers bfloat16 dtype with numpy
from pathlib import Path
from safetensors import safe_open

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# GGML types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12

# Block sizes
QK8_0 = 32
QK_K  = 256

# GGUF KV types
GGUF_VAL_UINT32 = 4
GGUF_VAL_STRING = 8


def bf16_to_f32(data_bytes: bytes, n_elements: int) -> np.ndarray:
    """Convert bfloat16 raw bytes to float32 numpy array."""
    bf16 = np.frombuffer(data_bytes, dtype=np.uint16).reshape(-1)
    # BF16 to F32: shift left by 16 bits
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)[:n_elements]


def quantize_f32_to_q8_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q8_0 format (32 elements per block)."""
    n = data.shape[0]
    assert n % QK8_0 == 0, f"Q8_0 requires n ({n}) divisible by {QK8_0}"
    n_blocks = n // QK8_0

    result = bytearray()
    for i in range(n_blocks):
        block = data[i * QK8_0 : (i + 1) * QK8_0]
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax > 0 else 0.0
        # Scale factor as fp16
        d_f16 = np.float16(d)
        id_val = 1.0 / float(d_f16) if float(d_f16) != 0 else 0.0
        # Quantize
        qs = np.clip(np.round(block * id_val), -128, 127).astype(np.int8)
        # Pack: [fp16 scale] [32 x int8]
        result += struct.pack('<e', float(d_f16))
        result += qs.tobytes()

    return bytes(result)


def quantize_f32_to_q4_K(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q4_K format (256 elements per super-block).

    Q4_K uses asymmetric min-max quantization with 8 sub-blocks of 32 elements.
    Each super-block: [fp16 d] [fp16 dmin] [12-byte packed scales/mins] [128-byte quants]
    """
    n = data.shape[0]
    assert n % QK_K == 0, f"Q4_K requires n ({n}) divisible by {QK_K}"
    n_blocks = n // QK_K

    result = bytearray()
    for i in range(n_blocks):
        block = data[i * QK_K : (i + 1) * QK_K]

        # Process 8 sub-blocks of 32 elements
        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)

        for j in range(8):
            sub = block[j * 32 : (j + 1) * 32]
            sub_min = np.min(sub)
            sub_max = np.max(sub)

            # Asymmetric: map [min, max] to [0, 15]
            if sub_max == sub_min:
                scales[j] = 0
                mins[j] = -sub_min
            else:
                scales[j] = (sub_max - sub_min) / 15.0
                mins[j] = -sub_min

        # Find super-block scale and min
        max_scale = np.max(scales)
        max_min = np.max(mins)

        if max_scale > 0:
            inv_max_scale = 63.0 / max_scale
        else:
            inv_max_scale = 0.0

        if max_min > 0:
            inv_max_min = 63.0 / max_min
        else:
            inv_max_min = 0.0

        d = np.float16(max_scale / 63.0)
        dmin = np.float16(max_min / 63.0)

        # Quantize scales and mins to 6-bit
        q_scales = np.zeros(8, dtype=np.uint8)
        q_mins = np.zeros(8, dtype=np.uint8)
        for j in range(8):
            q_scales[j] = min(63, int(np.round(scales[j] * inv_max_scale)))
            q_mins[j] = min(63, int(np.round(mins[j] * inv_max_min)))

        # Pack scales/mins into 12 bytes (K_SCALE_SIZE)
        # Layout from ggml: lower 4 bits in first 4 bytes, upper 2 bits packed later
        packed_scales = bytearray(12)
        # Bytes 0-3: lower nibbles of scales[0..3] | lower nibbles of mins[0..3]
        packed_scales[0] = (q_scales[0] & 0x3F) | ((q_mins[0] & 0x3F) << 6) & 0xFF
        packed_scales[1] = ((q_mins[0] >> 2) & 0x0F) | ((q_scales[1] & 0x3F) << 4) & 0xFF
        packed_scales[2] = ((q_scales[1] >> 4) & 0x03) | ((q_mins[1] & 0x3F) << 2) & 0xFF
        packed_scales[3] = (q_scales[2] & 0x3F) | ((q_mins[2] & 0x3F) << 6) & 0xFF
        packed_scales[4] = ((q_mins[2] >> 2) & 0x0F) | ((q_scales[3] & 0x3F) << 4) & 0xFF
        packed_scales[5] = ((q_scales[3] >> 4) & 0x03) | ((q_mins[3] & 0x3F) << 2) & 0xFF
        packed_scales[6] = (q_scales[4] & 0x3F) | ((q_mins[4] & 0x3F) << 6) & 0xFF
        packed_scales[7] = ((q_mins[4] >> 2) & 0x0F) | ((q_scales[5] & 0x3F) << 4) & 0xFF
        packed_scales[8] = ((q_scales[5] >> 4) & 0x03) | ((q_mins[5] & 0x3F) << 2) & 0xFF
        packed_scales[9] = (q_scales[6] & 0x3F) | ((q_mins[6] & 0x3F) << 6) & 0xFF
        packed_scales[10] = ((q_mins[6] >> 2) & 0x0F) | ((q_scales[7] & 0x3F) << 4) & 0xFF
        packed_scales[11] = ((q_scales[7] >> 4) & 0x03) | ((q_mins[7] & 0x3F) << 2) & 0xFF

        # Quantize the 256 elements to 4-bit
        d_val = float(d)
        dmin_val = float(dmin)
        qs = bytearray(QK_K // 2)  # 128 bytes

        for j in range(8):
            sc = d_val * q_scales[j]
            mn = dmin_val * q_mins[j]
            for l in range(32):
                idx = j * 32 + l
                val = block[idx]
                if sc > 0:
                    q = int(np.round((val + mn) / sc))
                else:
                    q = 0
                q = max(0, min(15, q))
                byte_idx = idx // 2
                if idx % 2 == 0:
                    qs[byte_idx] = q
                else:
                    qs[byte_idx] |= (q << 4)

        # Pack the super-block: [fp16 d] [fp16 dmin] [12B scales] [128B quants]
        result += struct.pack('<e', float(d))
        result += struct.pack('<e', float(dmin))
        result += bytes(packed_scales)
        result += bytes(qs)

    return bytes(result)


class GGUFWriter:
    """Minimal GGUF v3 writer."""

    def __init__(self):
        self.kv_pairs = []    # [(key, type, value_bytes)]
        self.tensors = []     # [(name, ndim, shape, type, data_bytes)]

    def add_u32(self, key: str, value: int):
        val_bytes = struct.pack('<I', value)
        self.kv_pairs.append((key, GGUF_VAL_UINT32, val_bytes))

    def add_string(self, key: str, value: str):
        encoded = value.encode('utf-8')
        val_bytes = struct.pack('<Q', len(encoded)) + encoded
        self.kv_pairs.append((key, GGUF_VAL_STRING, val_bytes))

    def add_tensor(self, name: str, shape: tuple, ggml_type: int, data: bytes):
        self.tensors.append((name, len(shape), shape, ggml_type, data))

    def write(self, path: str):
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.kv_pairs)))

            # KV pairs
            for key, kv_type, val_bytes in self.kv_pairs:
                key_enc = key.encode('utf-8')
                f.write(struct.pack('<Q', len(key_enc)))
                f.write(key_enc)
                f.write(struct.pack('<I', kv_type))
                f.write(val_bytes)

            # Tensor infos (name, ndim, shape, type, offset)
            # First pass: compute offsets
            offsets = []
            current_offset = 0
            for name, ndim, shape, ggml_type, data in self.tensors:
                # Align tensor data
                align = GGUF_ALIGNMENT
                current_offset = (current_offset + align - 1) & ~(align - 1)
                offsets.append(current_offset)
                current_offset += len(data)

            for idx, (name, ndim, shape, ggml_type, data) in enumerate(self.tensors):
                name_enc = name.encode('utf-8')
                f.write(struct.pack('<Q', len(name_enc)))
                f.write(name_enc)
                f.write(struct.pack('<I', ndim))
                for d in shape:
                    f.write(struct.pack('<Q', d))
                f.write(struct.pack('<I', ggml_type))
                f.write(struct.pack('<Q', offsets[idx]))

            # Align to GGUF_ALIGNMENT before data blob
            pos = f.tell()
            aligned_pos = (pos + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1)
            f.write(b'\x00' * (aligned_pos - pos))

            # Tensor data
            data_start = f.tell()
            for idx, (name, ndim, shape, ggml_type, data) in enumerate(self.tensors):
                target = data_start + offsets[idx]
                current = f.tell()
                if current < target:
                    f.write(b'\x00' * (target - current))
                f.write(data)


# Model weight name mappings: safetensors name -> (gguf_name, quantization)
# "enc" = encoder, "dec" = decoder

ENC_PREFIX = "thinker.audio_tower."
DEC_PREFIX = "thinker.model."


def load_safetensors(model_dir: str):
    """Load all safetensors files from model directory."""
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    tensors = {}
    for fpath in files:
        with safe_open(str(fpath), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        print(f"  Loaded {fpath.name}: {len([k for k in tensors])} tensors total")

    return tensors


def get_tensor_f32(tensors: dict, name: str) -> np.ndarray:
    """Get a tensor as float32, converting from bf16 if needed."""
    t = tensors[name]
    if t.dtype == np.float32:
        return t.flatten()
    elif t.dtype == np.float16:
        return t.astype(np.float32).flatten()
    elif hasattr(t.dtype, 'name') and 'bfloat16' in t.dtype.name:
        return t.astype(np.float32).flatten()
    else:
        # Try interpreting as uint16 (bf16 raw)
        raw = t.view(np.uint16).flatten()
        f32_bits = raw.astype(np.uint32) << 16
        return f32_bits.view(np.float32)


def load_config(model_dir: str) -> dict:
    """Load config.json from model directory."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gguf>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading config from {model_dir}/config.json...")
    config = load_config(model_dir)

    # Extract model parameters — handle nested config structure
    # Qwen3-ASR config: config["thinker_config"]["audio_config"] / ["text_config"]
    thinker = config.get("thinker_config", config)
    audio_cfg = thinker.get("audio_config", config.get("audio_config", config))
    text_cfg = thinker.get("text_config", config.get("text_config", config))

    # Encoder params
    enc_d_model = audio_cfg.get("d_model", audio_cfg.get("hidden_size", 896))
    enc_layers = audio_cfg.get("encoder_layers", audio_cfg.get("num_hidden_layers", 18))
    enc_heads = audio_cfg.get("encoder_attention_heads", audio_cfg.get("num_attention_heads", 14))
    enc_head_dim = enc_d_model // enc_heads  # typically 64
    enc_ffn_dim = audio_cfg.get("encoder_ffn_dim", audio_cfg.get("intermediate_size", 3584))
    enc_output_dim = audio_cfg.get("output_dim", 1024)

    # Decoder params
    dec_hidden = text_cfg.get("hidden_size", 1024)
    dec_layers = text_cfg.get("num_hidden_layers", 28)
    dec_heads = text_cfg.get("num_attention_heads", 16)
    dec_kv_heads = text_cfg.get("num_key_value_heads", 8)
    dec_head_dim = dec_hidden // dec_heads  # typically 128 (or from config)
    dec_head_dim = text_cfg.get("head_dim", dec_head_dim)
    dec_intermediate = text_cfg.get("intermediate_size", 3072)
    vocab_size = text_cfg.get("vocab_size", 151936)

    variant = "1.7B" if enc_layers > 18 else "0.6B"
    print(f"Model: Qwen3-ASR-{variant}")
    print(f"  Encoder: d_model={enc_d_model}, layers={enc_layers}, heads={enc_heads}, ffn={enc_ffn_dim}")
    print(f"  Decoder: hidden={dec_hidden}, layers={dec_layers}, heads={dec_heads}, kv_heads={dec_kv_heads}")
    print(f"  Decoder: head_dim={dec_head_dim}, intermediate={dec_intermediate}, vocab={vocab_size}")

    print(f"\nLoading safetensors from {model_dir}...")
    tensors = load_safetensors(model_dir)

    writer = GGUFWriter()

    # Write config as KV metadata
    writer.add_string("general.architecture", "qwen_asr")
    writer.add_u32("qwen_asr.enc_d_model", enc_d_model)
    writer.add_u32("qwen_asr.enc_layers", enc_layers)
    writer.add_u32("qwen_asr.enc_heads", enc_heads)
    writer.add_u32("qwen_asr.enc_head_dim", enc_head_dim)
    writer.add_u32("qwen_asr.enc_ffn_dim", enc_ffn_dim)
    writer.add_u32("qwen_asr.enc_output_dim", enc_output_dim)
    writer.add_u32("qwen_asr.dec_hidden", dec_hidden)
    writer.add_u32("qwen_asr.dec_layers", dec_layers)
    writer.add_u32("qwen_asr.dec_heads", dec_heads)
    writer.add_u32("qwen_asr.dec_kv_heads", dec_kv_heads)
    writer.add_u32("qwen_asr.dec_head_dim", dec_head_dim)
    writer.add_u32("qwen_asr.dec_intermediate", dec_intermediate)
    writer.add_u32("qwen_asr.vocab_size", vocab_size)

    n_tensors = 0

    # ---- Encoder Conv Stem (F32) ----
    print("\nProcessing encoder conv stem (F32)...")
    conv_map = [
        (f"{ENC_PREFIX}conv2d1.weight", "enc.conv1.weight"),
        (f"{ENC_PREFIX}conv2d1.bias",   "enc.conv1.bias"),
        (f"{ENC_PREFIX}conv2d2.weight", "enc.conv2.weight"),
        (f"{ENC_PREFIX}conv2d2.bias",   "enc.conv2.bias"),
        (f"{ENC_PREFIX}conv2d3.weight", "enc.conv3.weight"),
        (f"{ENC_PREFIX}conv2d3.bias",   "enc.conv3.bias"),
    ]
    for st_name, gguf_name in conv_map:
        data = get_tensor_f32(tensors, st_name)
        writer.add_tensor(gguf_name, (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

    # ---- Encoder Conv Output Projection (Q8_0) ----
    print("Processing encoder conv_out projection (Q8_0)...")
    data = get_tensor_f32(tensors, f"{ENC_PREFIX}conv_out.weight")
    assert len(data) % QK8_0 == 0, f"conv_out size {len(data)} not divisible by {QK8_0}"
    q8_data = quantize_f32_to_q8_0(data)
    writer.add_tensor("enc.conv_out.weight", (len(data),), GGML_TYPE_Q8_0, q8_data)
    n_tensors += 1

    # ---- Encoder Transformer Layers (Q8_0 weights, F32 biases/norms) ----
    print(f"Processing {enc_layers} encoder layers (Q8_0)...")
    for i in range(enc_layers):
        if (i + 1) % 6 == 0 or i == 0:
            print(f"  Encoder layer {i}/{enc_layers}...")

        lp = f"{ENC_PREFIX}layers.{i}"

        # Attention weights (Q8_0)
        for proj, gguf_proj in [("self_attn.q_proj", "attn.q"),
                                 ("self_attn.k_proj", "attn.k"),
                                 ("self_attn.v_proj", "attn.v"),
                                 ("self_attn.out_proj", "attn.o")]:
            # Weight -> Q8_0
            data = get_tensor_f32(tensors, f"{lp}.{proj}.weight")
            assert len(data) % QK8_0 == 0, f"Encoder {proj} weight size {len(data)} not divisible by {QK8_0}"
            q8_data = quantize_f32_to_q8_0(data)
            writer.add_tensor(f"enc.layers.{i}.{gguf_proj}.weight",
                            (len(data),), GGML_TYPE_Q8_0, q8_data)
            n_tensors += 1

            # Bias -> F32
            data = get_tensor_f32(tensors, f"{lp}.{proj}.bias")
            writer.add_tensor(f"enc.layers.{i}.{gguf_proj}.bias",
                            (len(data),), GGML_TYPE_F32, data.tobytes())
            n_tensors += 1

        # Attention LayerNorm (F32)
        for suffix in ["weight", "bias"]:
            data = get_tensor_f32(tensors, f"{lp}.self_attn_layer_norm.{suffix}")
            writer.add_tensor(f"enc.layers.{i}.attn_norm.{suffix}",
                            (len(data),), GGML_TYPE_F32, data.tobytes())
            n_tensors += 1

        # FFN weights (Q8_0) and biases (F32)
        for fc, gguf_fc in [("fc1", "ffn.fc1"), ("fc2", "ffn.fc2")]:
            data = get_tensor_f32(tensors, f"{lp}.{fc}.weight")
            assert len(data) % QK8_0 == 0, f"Encoder {fc} weight size {len(data)} not divisible by {QK8_0}"
            q8_data = quantize_f32_to_q8_0(data)
            writer.add_tensor(f"enc.layers.{i}.{gguf_fc}.weight",
                            (len(data),), GGML_TYPE_Q8_0, q8_data)
            n_tensors += 1

            data = get_tensor_f32(tensors, f"{lp}.{fc}.bias")
            writer.add_tensor(f"enc.layers.{i}.{gguf_fc}.bias",
                            (len(data),), GGML_TYPE_F32, data.tobytes())
            n_tensors += 1

        # FFN LayerNorm (F32)
        for suffix in ["weight", "bias"]:
            data = get_tensor_f32(tensors, f"{lp}.final_layer_norm.{suffix}")
            writer.add_tensor(f"enc.layers.{i}.ffn_norm.{suffix}",
                            (len(data),), GGML_TYPE_F32, data.tobytes())
            n_tensors += 1

    # ---- Encoder Post-Layers (F32 norms, Q8_0 projections) ----
    print("Processing encoder post-layers...")
    for suffix in ["weight", "bias"]:
        data = get_tensor_f32(tensors, f"{ENC_PREFIX}ln_post.{suffix}")
        writer.add_tensor(f"enc.ln_post.{suffix}", (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

    for proj_name, gguf_name in [("proj1", "enc.proj1"), ("proj2", "enc.proj2")]:
        data = get_tensor_f32(tensors, f"{ENC_PREFIX}{proj_name}.weight")
        assert len(data) % QK8_0 == 0
        q8_data = quantize_f32_to_q8_0(data)
        writer.add_tensor(f"{gguf_name}.weight", (len(data),), GGML_TYPE_Q8_0, q8_data)
        n_tensors += 1

        data = get_tensor_f32(tensors, f"{ENC_PREFIX}{proj_name}.bias")
        writer.add_tensor(f"{gguf_name}.bias", (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

    # ---- Decoder Token Embeddings ----
    print("Processing decoder token embeddings (F16 + Q4_K)...")
    embed_key = f"{DEC_PREFIX}embed_tokens.weight"
    embed_f32 = get_tensor_f32(tensors, embed_key)
    n_embed = vocab_size * dec_hidden

    # F16 copy for embedding lookup
    embed_f16 = embed_f32[:n_embed].astype(np.float32)
    embed_f16_data = np.array(embed_f16, dtype=np.float16).tobytes()
    writer.add_tensor("dec.tok_emb.f16", (n_embed,), GGML_TYPE_F16, embed_f16_data)
    n_tensors += 1

    # Q4_K copy for argmax
    assert n_embed % QK_K == 0, f"Embedding size {n_embed} not divisible by {QK_K}"
    q4k_data = quantize_f32_to_q4_K(embed_f32[:n_embed])
    writer.add_tensor("dec.tok_emb.q4k", (n_embed,), GGML_TYPE_Q4_K, q4k_data)
    n_tensors += 1

    # ---- Decoder Layers (Q4_K weights, F32 norms) ----
    print(f"Processing {dec_layers} decoder layers (Q4_K)...")
    for i in range(dec_layers):
        if (i + 1) % 7 == 0 or i == 0:
            print(f"  Decoder layer {i}/{dec_layers}...")

        lp = f"{DEC_PREFIX}layers.{i}"

        # Attention weights (Q4_K, no bias)
        for proj, gguf_proj in [("self_attn.q_proj", "attn.q"),
                                 ("self_attn.k_proj", "attn.k"),
                                 ("self_attn.v_proj", "attn.v"),
                                 ("self_attn.o_proj", "attn.o")]:
            data = get_tensor_f32(tensors, f"{lp}.{proj}.weight")
            assert len(data) % QK_K == 0, f"Decoder {proj} weight size {len(data)} not divisible by {QK_K}"
            q4k_data = quantize_f32_to_q4_K(data)
            writer.add_tensor(f"dec.layers.{i}.{gguf_proj}.weight",
                            (len(data),), GGML_TYPE_Q4_K, q4k_data)
            n_tensors += 1

        # Per-head Q/K RMSNorm (F32)
        data = get_tensor_f32(tensors, f"{lp}.self_attn.q_norm.weight")
        writer.add_tensor(f"dec.layers.{i}.attn.q_norm.weight",
                        (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

        data = get_tensor_f32(tensors, f"{lp}.self_attn.k_norm.weight")
        writer.add_tensor(f"dec.layers.{i}.attn.k_norm.weight",
                        (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

        # RMSNorm (F32)
        data = get_tensor_f32(tensors, f"{lp}.input_layernorm.weight")
        writer.add_tensor(f"dec.layers.{i}.input_norm.weight",
                        (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

        data = get_tensor_f32(tensors, f"{lp}.post_attention_layernorm.weight")
        writer.add_tensor(f"dec.layers.{i}.post_attn_norm.weight",
                        (len(data),), GGML_TYPE_F32, data.tobytes())
        n_tensors += 1

        # Gate+Up fusion: interleave rows [gate_row_0, up_row_0, gate_row_1, up_row_1, ...]
        gate_data = get_tensor_f32(tensors, f"{lp}.mlp.gate_proj.weight")
        up_data = get_tensor_f32(tensors, f"{lp}.mlp.up_proj.weight")
        gate_2d = gate_data.reshape(dec_intermediate, dec_hidden)
        up_2d = up_data.reshape(dec_intermediate, dec_hidden)
        # Interleave: [2*intermediate, hidden]
        fused = np.empty((2 * dec_intermediate, dec_hidden), dtype=np.float32)
        fused[0::2] = gate_2d
        fused[1::2] = up_2d
        fused_flat = fused.flatten()
        assert len(fused_flat) % QK_K == 0
        q4k_data = quantize_f32_to_q4_K(fused_flat)
        writer.add_tensor(f"dec.layers.{i}.mlp.gate_up.weight",
                        (len(fused_flat),), GGML_TYPE_Q4_K, q4k_data)
        n_tensors += 1

        # Down projection (Q4_K)
        data = get_tensor_f32(tensors, f"{lp}.mlp.down_proj.weight")
        assert len(data) % QK_K == 0
        q4k_data = quantize_f32_to_q4_K(data)
        writer.add_tensor(f"dec.layers.{i}.mlp.down.weight",
                        (len(data),), GGML_TYPE_Q4_K, q4k_data)
        n_tensors += 1

    # ---- Decoder Final Norm ----
    print("Processing decoder final norm...")
    data = get_tensor_f32(tensors, f"{DEC_PREFIX}norm.weight")
    writer.add_tensor("dec.norm.weight", (len(data),), GGML_TYPE_F32, data.tobytes())
    n_tensors += 1

    # ---- Write GGUF ----
    print(f"\nWriting {n_tensors} tensors to {output_path}...")
    writer.write(output_path)

    file_size = os.path.getsize(output_path)
    print(f"Done! Output: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
