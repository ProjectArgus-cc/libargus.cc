# Synthetic test fixtures

`tiny.gguf` is the existing generated 32-dimensional, one-layer text model.
`tiny-embed.gguf` is a synthetic 32-dimensional, one-layer embedding model configured with
MEAN pooling and non-causal attention for deterministic offline integration testing of
libargus bidirectional embeddings pipelines without external downloads.
`tiny-mmproj.gguf` is a compatible synthetic CLIP/MLP vision projector with
8-dimensional vision state, two layers (LLaVA selects the first), four image
patches and a 32-dimensional projection. It executes real upstream image
preprocessing, attention, projection, embedding decode and terminal-text logits.
It has no trained-model quality claim or external model weights.

`tiny-video.y4m` contains three distinct 16×16 luminance frames at 2 fps. The
mandatory video test decodes it through actual FFprobe/FFmpeg subprocesses.
Missing tools fail that lane. Iterator ownership/concurrency tests separately
use the test-only adapter and run on every supported OS.

Synthetic fixtures are generated with `export_test_model.py`, `export_test_embedding_model.py`,
and `export_test_projector.py`, distributed under this project's MIT license.
No external dataset, download or third-party weight license is required.
Regeneration must produce these exact SHA-256 values:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| tiny-embed.gguf | 115,104 | 0789433d2682a4678cfcf5d61d7f5366da22e218508dd7ebf748cf1de158b2f0 |
| tiny-mmproj.gguf | 17,920 | cd6155a72b3b5a0627db7d5b412824f5c4975f787e80272aa8293e173ad56d8a |
| tiny-video.y4m | 1,210 | 12b8f2799454def8c4a62f1420766b3585204ff5d5e0d68ead31387544f70f0c |
