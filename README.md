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

## Initialization

The SoundStream paper proposes that the codebook should be initialized by the kmeans centroids of the first batch. You can easily turn on this feature with one flag `kmeans_init = True`, for either `VectorQuantize` or `ResidualVQ` class

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 4,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = residual_vq(x)
```

## Increasing codebook usage

This repository will contain a few techniques from various papers to combat "dead" codebook entries, which is a common problem when using vector quantizers.

### Lower codebook dimension

The <a href="https://openreview.net/forum?id=pfNyExj7z2">Improved VQGAN paper</a> proposes to have the codebook kept in a lower dimension. The encoder values are projected down before being projected back to high dimensional after quantization. You can set this with the `codebook_dim` hyperparameter.

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    codebook_dim = 16      # paper proposes setting this to 32 or as low as 8 to increase codebook usage
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```

### Cosine similarity

The <a href="https://openreview.net/forum?id=pfNyExj7z2">Improved VQGAN paper</a> also proposes to l2 normalize the codes and the encoded vectors, which boils down to using cosine similarity for the distance. They claim enforcing the vectors on a sphere leads to improvements in code usage and downstream reconstruction. You can turn this on by setting `use_cosine_sim = True`

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    use_cosine_sim = True   # set this to True
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```

### Expiring stale codes

Finally, the SoundStream paper has a scheme where they replace codes that have hits below a certain threshold with randomly selected vector from the current batch. You can set this threshold with `threshold_ema_dead_code` keyword.

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 512,
    threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
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

```bibtex
@inproceedings{anonymous2022vectorquantized,
    title   = {Vector-quantized Image Modeling with Improved {VQGAN}},
    author  = {Anonymous},
    booktitle = {Submitted to The Tenth International Conference on Learning Representations },
    year    = {2022},
    url     = {https://openreview.net/forum?id=pfNyExj7z2},
    note    = {under review}
}
```
