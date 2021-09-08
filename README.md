## Vector Quantization - Pytorch

A vector quantization library originally transcribed from Deepmind's tensorflow implementation, made conveniently into a package. It uses exponential moving averages to update the dictionary.

VQ has been successfully used by Deepmind and OpenAI for high quality generation of images (VQ-VAE-2) and music (Jukebox).

## Install

```bash
$ pip install vector-quantize-pytorch
```

## Usage

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 512,     # codebook size
    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
    commitment = 1.          # the weight on the commitment loss
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x) # (1, 1024, 256), (1, 1024), (1)
```

## Variants

This <a href="https://arxiv.org/abs/2107.03312">paper</a> proposes to use multiple vector quantizers to recursively quantize the residuals of the waveform. You can use this with the `ResidualVQ` class and one extra initialization parameter.

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    num_quantizers = 8,      # specify number of quantizers
    codebook_size = 1024,    # codebook size
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = residual_vq(x)

# (1, 1024, 256), (8, 1, 1024), (8, 1)
# (batch, seq, dim), (quantizer, batch, seq), (quantizer, batch)
```

## Citations

```bibtex
@misc{oord2018neural,
    title   = {Neural Discrete Representation Learning},
    author  = {Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
    year    = {2018},
    eprint  = {1711.00937},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{zeghidour2021soundstream,
    title   = {SoundStream: An End-to-End Neural Audio Codec},
    author  = {Neil Zeghidour and Alejandro Luebs and Ahmed Omran and Jan Skoglund and Marco Tagliasacchi},
    year    = {2021},
    eprint  = {2107.03312},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```
