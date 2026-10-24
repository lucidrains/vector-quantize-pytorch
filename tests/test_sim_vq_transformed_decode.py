import pytest
import torch
from torch import nn

from vector_quantize_pytorch import SimVQ


def make_quantizer(transform_kind, channel_first):
    linear = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(3))
    if transform_kind == "batch_norm":
        normalizer = nn.BatchNorm1d(3)
        with torch.no_grad():
            normalizer.running_mean.copy_(torch.tensor([0.1, -0.3, 0.2]))
            normalizer.running_var.copy_(torch.tensor([1.2, 2.3, 0.7]))
    elif transform_kind == "row_softmax":
        normalizer = nn.Softmax(dim=0)
    else:
        normalizer = nn.Identity()
    model = SimVQ(
        dim=3,
        codebook_size=4,
        codebook_transform=nn.Sequential(linear, normalizer),
        channel_first=channel_first,
        rotation_trick=False,
    ).eval()
    with torch.no_grad():
        model.frozen_codebook.copy_(
            torch.tensor(
                [
                    [0.1, 1.2, -0.7],
                    [0.8, -0.5, 1.1],
                    [-0.9, 0.3, 0.4],
                    [1.5, 0.7, -1.3],
                ]
            )
        )
    return model


@pytest.mark.parametrize("transform_kind", ["linear", "batch_norm", "row_softmax"])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("image", [False, True])
def test_decode_matches_full_transformed_codebook_and_gradients(
    transform_kind, channel_first, image
):
    model = make_quantizer(transform_kind, channel_first)
    indices = torch.tensor([[0, 2], [3, 1]])
    if image:
        indices = torch.tensor([[[0, 1], [2, 3]], [[3, 2], [1, 0]]])
    # A code ID identifies one row of the complete transformed codebook.
    table = model.code_transform(model.frozen_codebook)
    expected = table[indices]
    if channel_first:
        expected = expected.movedim(-1, 1)
    actual = model.indices_to_codes(indices)
    torch.testing.assert_close(actual, expected)
    weight = model.code_transform[0].weight
    actual_grad = torch.autograd.grad(actual.square().sum(), weight)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)

    # Run the actual quantizer forward, using the selected table rows as inputs.
    inputs = expected.detach().clone().requires_grad_()
    quantized, encoded, loss = model(inputs)
    assert torch.equal(encoded, indices)
    torch.testing.assert_close(quantized, expected.detach())
    torch.testing.assert_close(model.indices_to_codes(encoded), quantized)
    (quantized.square().mean() + loss).backward()
    assert torch.isfinite(inputs.grad).all()


def test_decode_updates_batch_normalization_once():
    model = make_quantizer("batch_norm", False).train()
    normalizer = model.code_transform[1]
    # This shape is accepted by BatchNorm1d in either path, so the test observes
    # the extra state update independently of dimension-mismatch exceptions.
    indices = torch.tensor([[0, 1, 2], [1, 2, 3]])
    model.indices_to_codes(indices)
    assert normalizer.num_batches_tracked.item() == 1
