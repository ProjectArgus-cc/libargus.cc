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

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "tests/data/tiny-embed.gguf"
    generate_tiny_embedding_gguf(out_file)
