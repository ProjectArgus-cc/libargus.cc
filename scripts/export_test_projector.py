#!/usr/bin/env python3
"""Generate project-owned, deterministic F32 CLIP/MLP weights, without ML dependencies.

Two tiny vision layers (one selected by LLaVA) and a 32-dimensional output match
tests/data/tiny.gguf. These synthetic weights test execution, not model quality.
GGUF v3 integers, lengths, tensor dimensions and F32 values are little endian.
"""
import hashlib
import math
import struct
from pathlib import Path

def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data

def generate(destination):
    metadata = []
    def add(name, kind, value):
        formats = {4: '<I', 6: '<f', 7: '<?'}
        encoded = string(value) if kind == 8 else struct.pack(formats[kind], value)
        metadata.append(string(name) + struct.pack('<I', kind) + encoded)
    add('general.architecture', 8, 'clip')
    add('general.name', 8, 'Argus synthetic test projector')
    add('general.file_type', 4, 0)
    add('clip.projector_type', 8, 'mlp')
    add('clip.has_vision_encoder', 7, True)
    add('clip.has_audio_encoder', 7, False)
    add('clip.use_gelu', 7, True)
    for name, value in {'embedding_length': 8, 'feed_forward_length': 16,
                        'block_count': 2, 'projection_dim': 32,
                        'attention.head_count': 2, 'image_size': 16, 'patch_size': 8}.items():
        add('clip.vision.' + name, 4, value)
    add('clip.vision.attention.layer_norm_epsilon', 6, 1e-5)
    for name, value in [('image_mean', 0.5), ('image_std', 0.5)]:
        metadata.append(string('clip.vision.' + name) + struct.pack('<IIQ3f', 9, 6, 3, value, value, value))
    tensors = []
    def tensor(name, shape, norm=False):
        # Integer arithmetic + exact binary fractions avoid PRNG/library drift.
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        count = math.prod(shape)
        values = [1.0 if norm else (((seed + i * 2654435761) & 255) - 128) / 8192 for i in range(count)]
        tensors.append((name, shape, struct.pack('<' + 'f' * count, *values)))
    tensor('v.patch_embd.weight', (8, 8, 3, 8))
    tensor('v.position_embd.weight', (8, 4))
    for layer in range(2):
        prefix = f'v.blk.{layer}.'
        for name in ('attn_q', 'attn_k', 'attn_v', 'attn_out'):
            tensor(prefix + name + '.weight', (8, 8))
        for name in ('ln1', 'ln2'):
            tensor(prefix + name + '.weight', (8,), norm=True)
        tensor(prefix + 'ffn_up.weight', (8, 16))
        tensor(prefix + 'ffn_down.weight', (16, 8))
    tensor('mm.0.weight', (8, 32)); tensor('mm.0.bias', (32,))
    tensor('mm.2.weight', (32, 32)); tensor('mm.2.bias', (32,))
    headers, payload, offset = [], [], 0
    for name, shape, data in tensors:
        headers.append(string(name) + struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', 0, offset))
        padding = bytes((-len(data)) % 32)
        payload.extend((data, padding)); offset += len(data) + len(padding)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata)) + b''.join(metadata + headers)
    destination.write_bytes(header + bytes((-len(header)) % 32) + b''.join(payload))

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'tests/data'
    generate(root / 'tiny-mmproj.gguf')
    # Three distinguishable frames, lossless YUV4MPEG2; decoded by real FFmpeg.
    video = b'YUV4MPEG2 W16 H16 F2:1 Ip A1:1 C420jpeg\n'
    for y in (32, 128, 224):
        video += b'FRAME\n' + bytes([y]) * 256 + bytes([128]) * 128
    (root / 'tiny-video.y4m').write_bytes(video)
    for name in ('tiny-mmproj.gguf', 'tiny-video.y4m'):
        path = root / name
        print(hashlib.sha256(path.read_bytes()).hexdigest(), name, path.stat().st_size)
