# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange

from vector_quantize_pytorch import LFQ

lr = 5e-4
batch_size = 256
train_iter = 1000
seed = 1234
codebook_size = 2 ** 8
# 32 codes per image
num_codebooks = 32
entropy_loss_weight = 0.01
commitment_loss_weight = 0.25
diversity_gamma = 1.
device = "cuda" if torch.cuda.is_available() else "cpu"

class LFQAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim, 
        **vq_kwargs
    ):
        super().__init__()

        self.encode = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
        )

        self.lfq = LFQ(**vq_kwargs)

        self.decode = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
        )
        return

    def forward(self, x, mask=None):
        x = self.encode(x)
        x, indices, entropy_aux_loss = self.lfq(x, mask=mask)
        x = self.decode(x)
        return x, indices, entropy_aux_loss


def train(model, train_loader, train_iterations=1000, add_masked_data=False):
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

        og_shape = x.shape
        x = rearrange(x, 'b c h w -> b 1 (c h w)')

        mask = torch.ones(x.shape[0], 2 if add_masked_data else 1, dtype=torch.bool, device=x.device)
        if add_masked_data:
            masked_data = torch.randn_like(x)
            x = torch.concat([x,masked_data], dim=1)
            # Mask where masked_data is False
            mask[:,1] = False

        out, indices, entropy_aux_loss = model(x, mask=mask)

        rec_loss = F.l1_loss(out[mask], x[mask])
        (rec_loss + entropy_aux_loss).backward()

        opt.step()
        pbar.set_description(
              f"rec loss: {rec_loss.item():.3f} | "
            + f"entropy aux loss: {entropy_aux_loss.item():.3f} | "
            + f"active %: {indices[mask].unique().numel() / codebook_size * 100:.3f}"
        )
    return

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)


torch.random.manual_seed(seed)

mnist_h, mnist_w = 28, 28
mnist_c = 1
input_dim = mnist_h * mnist_w * mnist_c
# this is also the number of codes
hidden_dim = codebook_size

def get_model_and_opt():
    model = LFQAutoEncoder(
        input_dim,
        hidden_dim,
        dim=hidden_dim,
        codebook_size = codebook_size,
        entropy_loss_weight = entropy_loss_weight,
        diversity_gamma = diversity_gamma,
        commitment_loss_weight=commitment_loss_weight,
        num_codebooks=num_codebooks,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, opt

print("baseline")
model, opt = get_model_and_opt()
train(model, train_dataset, train_iterations=train_iter)

print("with masking")
model, opt = get_model_and_opt()
train(model, train_dataset, train_iterations=train_iter, add_masked_data=True)
