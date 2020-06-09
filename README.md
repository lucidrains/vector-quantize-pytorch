## Vector Quantization, in Pytorch

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
	n_embed = 512,     # size of the dictionary
	decay = 0.8, 	   # the exponential moving average decay, lower means the dictionary will change faster
	commitment = 1.    # the weight on the commitment loss
)

x = torch.randn(1, 1024, 256)
quantized, commit_loss = vq(x) # (1, 1024, 256), (1)
```
