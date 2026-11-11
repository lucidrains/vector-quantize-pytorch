from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from vector_quantize_pytorch import GroupedResidualVQ, ResidualVQ


def supervised_reference(vq, x, targets):
    """Compute greedy residual codes and label NLL without calling a quantizer."""
    projected = vq.project_in(x)
    codes = []
    losses = []

    for depth, layer in enumerate(vq.layers):
        previous = sum(codes, torch.zeros_like(projected))
        residual = projected - previous.detach()
        residual = residual - vq.quant_grad_frac * (previous - previous.detach())
        codebook = layer.codebook
        transformed = vq.implicit_neural_codebook and depth > 0
        if transformed:
            codebook = vq.mlps[depth - 1](codebook, condition = previous)

        if layer.use_cosine_sim:
            logits = F.normalize(residual, dim = -1) @ codebook.T
        else:
            differences = residual.unsqueeze(-2) - codebook
            if transformed:
                differences = differences + 1e-6  # pairwise_distance's epsilon
            logits = -torch.linalg.vector_norm(differences, dim = -1)

        target_logits = logits.gather(-1, targets[..., depth, None]).squeeze(-1)
        losses.append((logits.logsumexp(-1) - target_logits).mean())
        selected = logits.argmax(-1)
        if transformed:
            selected = selected[..., None, None].expand(
                *selected.shape, 1, codebook.size(-1)
            )
            codes.append(codebook.gather(-2, selected).squeeze(-2))
        else:
            codes.append(codebook[selected])

    return vq.project_out(torch.stack(codes).sum(0)), sum(losses)


def test_supervised_residual_vq_known_codes():
    vq = ResidualVQ(dim = 2, num_quantizers = 3, codebook_size = 3).eval()
    with torch.no_grad():
        for layer, scale in zip(vq.layers, (2., 1., 0.25)):
            layer.codebook = torch.tensor([[0., 0.], [scale, 0.], [0., scale]])

    x = torch.tensor([[[2.8, 0.2], [0.2, 2.8]]])
    targets = torch.tensor([[[0, 0, 1], [1, 1, 2]]])
    expected, expected_loss = supervised_reference(vq, x, targets)
    quantized, loss = vq(x, indices = targets)

    # The supplied labels supervise CE; they do not replace greedy code selection.
    torch.testing.assert_close(expected, torch.tensor([[[3., 0.25], [0.25, 3.]]]))
    torch.testing.assert_close(quantized, expected)
    torch.testing.assert_close(loss, expected_loss)
    assert loss.ndim == 0


@pytest.mark.parametrize('training', (True, False))
@pytest.mark.parametrize('shared_codebook', (True, False))
@pytest.mark.parametrize('dim', (2, 4))
@pytest.mark.parametrize('quant_grad_frac', (0., 0.25))
@pytest.mark.parametrize('single_token', (True, False))
def test_supervised_residual_vq_values_and_gradients(
    training, shared_codebook, dim, quant_grad_frac, single_token
):
    torch.manual_seed(42)
    vq = ResidualVQ(
        dim = dim,
        codebook_dim = 2,
        num_quantizers = 3,
        codebook_size = 5,
        shared_codebook = shared_codebook,
        learnable_codebook = True,
        ema_update = False,
        route_gradients_to_input = False,
        quant_grad_frac = quant_grad_frac,
        quantize_dropout = True
    ).train(training)
    reference = deepcopy(vq)
    shape = (2, dim) if single_token else (2, 4, dim)
    x = torch.randn(*shape, requires_grad = True)
    reference_x = x.detach().clone().requires_grad_()
    targets = torch.randint(0, 5, (*shape[:-1], 3))

    expected, expected_loss = supervised_reference(reference, reference_x, targets)
    quantized, loss = vq(x, indices = targets)

    assert quantized.shape == x.shape
    assert loss.ndim == 0
    torch.testing.assert_close(quantized, expected)
    torch.testing.assert_close(loss, expected_loss)

    (loss + quantized.square().mean()).backward()
    (expected_loss + expected.square().mean()).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol = 2e-6, rtol = 2e-5)
    for parameter, reference_parameter in zip(vq.parameters(), reference.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, atol = 2e-6, rtol = 2e-5
        )


@pytest.mark.parametrize('rotation_trick', (True, False))
def test_supervised_residual_vq_routes_output_gradients(rotation_trick):
    torch.manual_seed(46)
    vq = ResidualVQ(
        dim = 4,
        codebook_dim = 2,
        num_quantizers = 3,
        codebook_size = 5,
        ema_update = False,
        rotation_trick = rotation_trick
    )
    reference = deepcopy(vq)
    x = torch.randn(2, 4, 4, requires_grad = True)
    reference_x = x.detach().clone().requires_grad_()
    targets = torch.randint(0, 5, (2, 4, 3))

    _, expected_loss = supervised_reference(reference, reference_x, targets)
    expected_output, _, _ = reference(reference_x)
    quantized, loss = vq(x, indices = targets)

    torch.testing.assert_close(quantized, expected_output)
    torch.testing.assert_close(loss, expected_loss)
    (loss + quantized.square().mean()).backward()
    (expected_loss + expected_output.square().mean()).backward()

    torch.testing.assert_close(x.grad, reference_x.grad, atol = 2e-6, rtol = 2e-5)
    for parameter, reference_parameter in zip(vq.parameters(), reference.parameters()):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, atol = 2e-6, rtol = 2e-5
        )


@pytest.mark.parametrize('training', (True, False))
@pytest.mark.parametrize('implicit_neural_codebook', (True, False))
def test_supervised_residual_vq_codebook_variants(training, implicit_neural_codebook):
    torch.manual_seed(47)
    vq = ResidualVQ(
        dim = 2,
        num_quantizers = 3,
        codebook_size = 5,
        implicit_neural_codebook = implicit_neural_codebook,
        mlp_kwargs = {'depth': 1},
        use_cosine_sim = not implicit_neural_codebook,
        route_gradients_to_input = False,
        ema_update = False
    ).train(training)
    reference = deepcopy(vq)
    x = torch.randn(2, 4, 2, requires_grad = True)
    reference_x = x.detach().clone().requires_grad_()
    targets = torch.randint(0, 5, (2, 4, 3))
    expected, expected_loss = supervised_reference(reference, reference_x, targets)

    quantized, loss = vq(x, indices = targets)

    torch.testing.assert_close(quantized, expected)
    torch.testing.assert_close(loss, expected_loss)
    (loss + quantized.square().mean()).backward()
    (expected_loss + expected.square().mean()).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol = 2e-6, rtol = 2e-5)
    for parameter, reference_parameter in zip(vq.parameters(), reference.parameters()):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, atol = 2e-6, rtol = 2e-5
        )


@pytest.mark.parametrize('training', (True, False))
@pytest.mark.parametrize('return_zeros_for_masked_padding', (True, False))
def test_supervised_residual_vq_masked_output(
    training, return_zeros_for_masked_padding
):
    torch.manual_seed(43)
    vq = ResidualVQ(
        dim = 4,
        codebook_dim = 2,
        num_quantizers = 3,
        codebook_size = 5,
        ema_update = False,
        return_zeros_for_masked_padding = return_zeros_for_masked_padding
    ).train(training)
    x = torch.randn(2, 4, 4)
    targets = torch.randint(0, 5, (2, 4, 3))
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    expected, expected_loss = supervised_reference(vq, x, targets)
    padding = torch.zeros_like(x) if return_zeros_for_masked_padding else x

    quantized, loss = vq(x, indices = targets, mask = mask)

    expected = torch.where(mask[..., None], expected, padding)
    torch.testing.assert_close(quantized, expected)
    # Preserve VectorQuantize's existing supervised contract for valid targets:
    # mask controls padding and codebook updates, not the supplied CE labels.
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize('training', (True, False))
@pytest.mark.parametrize('shared_codebook', (True, False))
@pytest.mark.parametrize('freeze_codebook', (True, False))
@pytest.mark.parametrize('masked', (True, False))
def test_supervised_residual_vq_preserves_codebook_updates(
    training, shared_codebook, freeze_codebook, masked
):
    torch.manual_seed(44)
    vq = ResidualVQ(
        dim = 4,
        codebook_dim = 2,
        num_quantizers = 3,
        codebook_size = 5,
        shared_codebook = shared_codebook
    ).train(training)
    ordinary = deepcopy(vq)
    before = deepcopy(vq.state_dict())
    x = torch.randn(2, 4, 4)
    targets = torch.randint(0, 5, (2, 4, 3))
    mask = None
    if masked:
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    for _ in range(2):
        quantized, loss = vq(
            x, indices = targets, mask = mask, freeze_codebook = freeze_codebook
        )
        expected, _, _ = ordinary(x, mask = mask, freeze_codebook = freeze_codebook)
        torch.testing.assert_close(quantized, expected)
        assert torch.isfinite(loss)
        for name, value in vq.state_dict().items():
            torch.testing.assert_close(value, ordinary.state_dict()[name])

    if training and not freeze_codebook:
        assert not torch.equal(
            before['layers.0._codebook.embed_avg'], vq.layers[0]._codebook.embed_avg
        )
    else:
        for name, value in vq.state_dict().items():
            torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize('training', (True, False))
@pytest.mark.parametrize('shared_codebook', (True, False))
@pytest.mark.parametrize('codebook_dim', (2, 3))
def test_supervised_grouped_residual_vq(training, shared_codebook, codebook_dim):
    torch.manual_seed(45)
    vq = GroupedResidualVQ(
        dim = 6,
        groups = 2,
        codebook_dim = codebook_dim,
        num_quantizers = 3,
        codebook_size = 5,
        shared_codebook = shared_codebook,
        ema_update = False,
        quantize_dropout = True
    ).train(training)
    x = torch.randn(2, 4, 6)
    targets = torch.randint(0, 5, (2, 2, 4, 3))
    expected = [
        supervised_reference(rvq, chunk, group_targets)
        for rvq, chunk, group_targets in zip(vq.rvqs, x.chunk(2, dim = -1), targets)
    ]

    quantized, loss = vq(x, indices = targets)

    expected_output = torch.cat([out for out, _ in expected], dim = -1)
    torch.testing.assert_close(quantized, expected_output)
    torch.testing.assert_close(loss, sum(group_loss for _, group_loss in expected))
    assert loss.ndim == 0


def test_supervised_grouped_residual_vq_single_token():
    vq = GroupedResidualVQ(
        dim = 4, groups = 2, num_quantizers = 3, codebook_size = 5
    ).eval()
    x = torch.randn(2, 4)
    targets = torch.randint(0, 5, (2, 2, 3))
    expected = [
        supervised_reference(rvq, chunk, group_targets)
        for rvq, chunk, group_targets in zip(vq.rvqs, x.chunk(2, dim = -1), targets)
    ]

    quantized, loss = vq(x, indices = targets)

    expected_output = torch.cat([out for out, _ in expected], dim = -1)
    torch.testing.assert_close(quantized, expected_output)
    torch.testing.assert_close(loss, sum(group_loss for _, group_loss in expected))


@pytest.mark.parametrize('training', (True, False))
@pytest.mark.parametrize('forward_override', (True, False))
def test_supervised_residual_vq_rejects_beam_search_without_updates(
    training, forward_override
):
    vq = ResidualVQ(
        dim = 2,
        num_quantizers = 3,
        codebook_size = 5,
        beam_size = None if forward_override else 2
    ).train(training)
    before = deepcopy(vq.state_dict())
    x = torch.randn(2, 4, 2)
    targets = torch.zeros(2, 4, 3, dtype = torch.long)
    kwargs = {'beam_size': 2} if forward_override else {}

    with pytest.raises(AssertionError, match = 'beam search.*indices'):
        vq(x, indices = targets, **kwargs)

    for name, value in vq.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_supervised_residual_vq_can_disable_configured_beam_search():
    vq = ResidualVQ(
        dim = 2, num_quantizers = 3, codebook_size = 5, beam_size = 2
    ).eval()
    x = torch.randn(2, 4, 2)
    targets = torch.zeros(2, 4, 3, dtype = torch.long)
    expected, expected_loss = supervised_reference(vq, x, targets)

    quantized, loss = vq(x, indices = targets, beam_size = 1)

    torch.testing.assert_close(quantized, expected)
    torch.testing.assert_close(loss, expected_loss)


def test_supervised_residual_vq_respects_eval_beam_size():
    vq = ResidualVQ(
        dim = 2, num_quantizers = 3, codebook_size = 5,
        beam_size = 1, eval_beam_size = 2
    )
    x = torch.randn(2, 4, 2)
    targets = torch.zeros(2, 4, 3, dtype = torch.long)
    quantized, loss = vq(x, indices = targets)
    assert quantized.shape == x.shape
    assert loss.ndim == 0

    with pytest.raises(AssertionError, match = 'beam search.*indices'):
        vq.eval()(x, indices = targets)


def test_supervised_residual_vq_still_rejects_dropped_indices():
    vq = ResidualVQ(dim = 2, num_quantizers = 3, codebook_size = 5)
    x = torch.randn(2, 4, 2)
    targets = torch.full((2, 4, 3), -1, dtype = torch.long)

    with pytest.raises(AssertionError, match = 'indices were dropped out'):
        vq(x, indices = targets)
