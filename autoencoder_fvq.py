# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD

from vector_quantize_pytorch import VectorQuantize, Sequential

lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234
rotation_trick = True
device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class VQBridgeViT(Module):
    def __init__(
        self,
        dim,
        depth,
        input_dim = 32,
        patch_size = 16,
        dim_head = 16,
        heads = 4,
        add_residual = False
    ):
        super().__init__()
        self.add_residual = add_residual

        patch_dim = input_dim * patch_size
        self.patch_to_tokens = nn.Sequential(
            Rearrange('b (n p) c -> b n (p c)', p = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim = dim, dim_head = dim_head, heads = heads, depth = depth, mlp_dim = dim * 4)

        self.tokens_to_patch = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b n (p c) -> b (n p) c', p = patch_size),
        )

    def forward(self, x):
        residual = x

        x = self.patch_to_tokens(x)

        x = self.transformer(x)

        x = self.tokens_to_patch(x)

        if self.add_residual:
            x = x + residual

        return x

def SimpleVQAutoEncoder(**vq_kwargs):
    return Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.GELU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        VectorQuantize(dim=32, accept_image_fmap = True, **vq_kwargs),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        nn.GELU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
    )

def train(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))

        out, indices, cmt_loss = model(x)
        out = out.clamp(-1., 1.)

        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

torch.random.manual_seed(seed)

model = SimpleVQAutoEncoder(
    codebook_size = num_codes,
    learnable_codebook = True,
    in_place_codebook_optimizer = lambda *args, **kwargs: SGD(*args, **kwargs, lr = 1e-3),
    ema_update = False,
    # vq_bridge = None,
    vq_bridge = VQBridgeViT(
        dim = 256,
        input_dim = 32,
        patch_size = 2,
        depth = 1,
        add_residual = False
    )
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)
