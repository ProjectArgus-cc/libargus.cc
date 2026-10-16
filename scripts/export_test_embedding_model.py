#!/usr/bin/env python3
"""
Generates a minimal valid embedding LLaMA GGUF model for deterministic unit testing in libargus.
Architecture: 1 layer, 32 embedding dimension, 2 attention heads, 64 context length,
260 vocabulary items (control tokens + full 256 byte tokens for universal string tokenization),
MEAN pooling (llama.pooling_type = 1) and non-causal attention (llama.attention.causal = False).
File size: ~115KB.
"""

import hashlib
import os
import sys
import numpy as np

# Ensure vendored gguf is discovered if run directly
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
vendored_gguf = os.path.join(root_dir, "build/_deps/llamacpp-src/gguf-py")
if os.path.exists(vendored_gguf) and vendored_gguf not in sys.path:
    sys.path.insert(0, vendored_gguf)

pipx_gguf = os.path.expanduser("~/.local/share/pipx/venvs/gguf/lib/python3.14/site-packages")
if os.path.exists(pipx_gguf) and pipx_gguf not in sys.path:
    sys.path.insert(0, pipx_gguf)

import gguf

def generate_tiny_embedding_gguf(output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n_embd = 32
    n_head = 2
    n_head_kv = 2
    n_layer = 1
    n_ff = 64
    n_ctx = 512

    # Vocab: control tokens + full 256 byte tokens
    tokens = [b"<unk>", b"<s>", b"</s>", b"<pad>"]
    toktypes = [
        int(gguf.TokenType.UNKNOWN),
        int(gguf.TokenType.CONTROL),
        int(gguf.TokenType.CONTROL),
        int(gguf.TokenType.CONTROL)
    ]
    scores = [0.0, 0.0, 0.0, 0.0]

    for b in range(256):
        tokens.append(f"<0x{b:02X}>".encode("utf-8"))
        toktypes.append(int(gguf.TokenType.BYTE))
        scores.append(0.0)

    vocab_size = len(tokens) # 260 tokens

    writer = gguf.GGUFWriter(output_path, "llama")

    # Model architecture metadata
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_layer_norm_rms_eps(1e-5)

    # Embedding specific metadata: Mean pooling & non-causal attention
    writer.add_uint32("llama.pooling_type", 1) # LLAMA_POOLING_TYPE_MEAN
    writer.add_bool("llama.attention.causal", False)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(3)

    # Deterministic tensor weights
    rng = np.random.RandomState(42)

    def add_tensor(name, shape):
        data = (rng.randn(*shape) * 0.02).astype(np.float32)
        writer.add_tensor(name, data)

    # Embeddings & Output
    add_tensor("token_embd.weight", (vocab_size, n_embd))
    add_tensor("output_norm.weight", (n_embd,))
    add_tensor("output.weight", (vocab_size, n_embd))

    # Single transformer layer
    add_tensor("blk.0.attn_norm.weight", (n_embd,))
    add_tensor("blk.0.attn_q.weight", (n_embd, n_embd))
    add_tensor("blk.0.attn_k.weight", (n_embd, n_embd))
    add_tensor("blk.0.attn_v.weight", (n_embd, n_embd))
    add_tensor("blk.0.attn_output.weight", (n_embd, n_embd))
    add_tensor("blk.0.ffn_norm.weight", (n_embd,))
    add_tensor("blk.0.ffn_gate.weight", (n_ff, n_embd))
    add_tensor("blk.0.ffn_up.weight", (n_ff, n_embd))
    add_tensor("blk.0.ffn_down.weight", (n_embd, n_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sha256 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    print(f"Generated minimal embedding GGUF model: {output_path} ({os.path.getsize(output_path)} bytes, sha256={sha256})")


def generate_tiny_gemma_embedding2_gguf(output_path: str):
    """
    Generates a minimal valid EmbeddingGemma2 (gemma-embedding2) GGUF model for unit testing.
    Architecture: 2 layers (1 SWA + 1 dense via sliding_window_pattern=2), n_embd = 32 (backbone),
    n_embd_out = 16 (projected output), n_embd_per_layer = 8, 2 attention heads, MEAN pooling,
    non-causal no-KV-cache attention.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    arch = "gemma-embedding2"
    n_embd = 32
    n_embd_out = 16
    n_embd_per_layer = 8
    n_head = 2
    n_head_kv = 2
    n_embd_head = n_embd // n_head  # 16
    n_layer = 2
    n_ff = 64
    n_ctx = 512
    n_swa = 64

    tokens = [b"<unk>", b"<s>", b"</s>", b"<pad>"]
    toktypes = [
        int(gguf.TokenType.UNKNOWN),
        int(gguf.TokenType.CONTROL),
        int(gguf.TokenType.CONTROL),
        int(gguf.TokenType.CONTROL),
    ]
    scores = [0.0, 0.0, 0.0, 0.0]

    for b in range(256):
        tokens.append(f"<0x{b:02X}>".encode("utf-8"))
        toktypes.append(int(gguf.TokenType.BYTE))
        scores.append(0.0)

    vocab_size = len(tokens)  # 260 tokens

    writer = gguf.GGUFWriter(output_path, arch)

    # Model architecture metadata
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_uint32(f"{arch}.embedding_length_out", n_embd_out)
    writer.add_uint32(f"{arch}.embedding_length_per_layer_input", n_embd_per_layer)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(n_embd_head)
    writer.add_value_length(n_embd_head)
    writer.add_uint32(f"{arch}.attention.key_length_swa", n_embd_head)
    writer.add_uint32(f"{arch}.attention.value_length_swa", n_embd_head)
    writer.add_uint32(f"{arch}.attention.sliding_window", n_swa)
    writer.add_uint32(f"{arch}.attention.sliding_window_pattern", 2)
    writer.add_rope_dimension_count(n_embd_head)
    writer.add_uint32(f"{arch}.rope.dimension_count_swa", n_embd_head)
    writer.add_layer_norm_rms_eps(1e-5)

    # Embedding specific metadata: Mean pooling & non-causal attention
    writer.add_uint32(f"{arch}.pooling_type", 1)  # LLAMA_POOLING_TYPE_MEAN
    writer.add_bool(f"{arch}.attention.causal", False)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(3)

    rng = np.random.RandomState(42)

    def add_tensor(name, shape, scale=0.05):
        data = (rng.randn(*shape) * scale).astype(np.float32)
        writer.add_tensor(name, data)

    def add_ones(name, shape):
        data = np.ones(shape, dtype=np.float32)
        writer.add_tensor(name, data)

    # Note: GGUFWriter.add_tensor reverses numpy shape (ne1, ne0) -> ggml {ne0, ne1}
    add_tensor("token_embd.weight", (vocab_size, n_embd))
    add_tensor("per_layer_model_proj.weight", (n_embd_per_layer * n_layer, n_embd))
    add_ones("per_layer_proj_norm.weight", (n_embd_per_layer,))
    add_ones("output_norm.weight", (n_embd,))
    add_tensor("output.weight", (n_embd_out, n_embd))

    # 2 GemmaEmbedding2 transformer layers (blk.0 = SWA, blk.1 = dense)
    for il in range(n_layer):
        add_ones(f"blk.{il}.attn_norm.weight", (n_embd,))
        add_tensor(f"blk.{il}.attn_q.weight", (n_embd, n_embd))
        add_tensor(f"blk.{il}.attn_k.weight", (n_embd, n_embd))
        add_tensor(f"blk.{il}.attn_v.weight", (n_embd, n_embd))
        add_tensor(f"blk.{il}.attn_output.weight", (n_embd, n_embd))
        add_ones(f"blk.{il}.attn_q_norm.weight", (n_embd_head,))
        add_ones(f"blk.{il}.attn_k_norm.weight", (n_embd_head,))
        add_ones(f"blk.{il}.post_attention_norm.weight", (n_embd,))

        add_ones(f"blk.{il}.ffn_norm.weight", (n_embd,))
        add_tensor(f"blk.{il}.ffn_gate.weight", (n_ff, n_embd))
        add_tensor(f"blk.{il}.ffn_up.weight", (n_ff, n_embd))
        add_tensor(f"blk.{il}.ffn_down.weight", (n_embd, n_ff))
        add_ones(f"blk.{il}.post_ffw_norm.weight", (n_embd,))

        add_tensor(f"blk.{il}.inp_gate.weight", (n_embd_per_layer, n_embd))
        add_tensor(f"blk.{il}.proj.weight", (n_embd, n_embd_per_layer))
        add_ones(f"blk.{il}.post_norm.weight", (n_embd,))
        add_ones(f"blk.{il}.layer_output_scale.weight", (1,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sha256 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    print(f"Generated minimal GemmaEmbedding2 GGUF model: {output_path} ({os.path.getsize(output_path)} bytes, sha256={sha256})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gemma-embed2":
        out_file = sys.argv[2] if len(sys.argv) > 2 else "tests/data/tiny-gemma-embed2.gguf"
        generate_tiny_gemma_embedding2_gguf(out_file)
    else:
        out_file = sys.argv[1] if len(sys.argv) > 1 else "tests/data/tiny-embed.gguf"
        generate_tiny_embedding_gguf(out_file)
        if len(sys.argv) == 1:
            generate_tiny_gemma_embedding2_gguf("tests/data/tiny-gemma-embed2.gguf")

