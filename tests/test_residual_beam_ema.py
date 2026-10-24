import random

import pytest
import torch
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ


def selected_residual_statistics(model, x, indices, codebooks):
    residual = model.project_in(x).detach()
    statistics = []
    for stage, codebook in enumerate(codebooks):
        selected = indices[..., stage]
        valid = selected >= 0
        if not valid.any():
            break
        vectors = (
            F.normalize(residual, dim=-1)
            if model.layers[stage].use_cosine_sim
            else residual
        )
        counts = torch.stack(
            [(valid & (selected == code)).sum() for code in range(codebook.shape[1])]
        )
        sums = torch.stack(
            [
                vectors[valid & (selected == code)].sum(0)
                for code in range(codebook.shape[1])
            ]
        )
        statistics.append((counts[None].float(), sums[None].float()))
        codes = F.embedding(selected.clamp_min(0), codebook[0])
        residual = residual - codes.masked_fill(~valid[..., None], 0)
    return statistics


@pytest.mark.parametrize("shared_codebook", [False, True])
@pytest.mark.parametrize("use_cosine_sim", [False, True])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("dropout", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_beam_updates_selected_residuals_for_each_active_stage(
    shared_codebook, use_cosine_sim, projected, dropout, masked
):
    torch.manual_seed(40)
    model = ResidualVQ(
        dim=4 if projected else 2,
        codebook_dim=2,
        codebook_size=4,
        num_quantizers=3,
        beam_size=2,
        decay=0.6,
        shared_codebook=shared_codebook,
        use_cosine_sim=use_cosine_sim,
        quantize_dropout=dropout,
        kmeans_init=False,
        threshold_ema_dead_code=0,
        rotation_trick=False,
    ).train()
    x = torch.randn(2, 5, model.project_in.in_features if projected else 2)
    mask = (
        torch.tensor(
            [[True, True, False, True, False], [True, False, True, True, True]]
        )
        if masked
        else None
    )
    codebooks = [layer._codebook.embed.detach().clone() for layer in model.layers]
    initial_counts = [layer._codebook.cluster_size.clone() for layer in model.layers]
    initial_sums = [layer._codebook.embed_avg.clone() for layer in model.layers]
    _, indices, _ = model(x, mask=mask, rand_quantize_dropout_fixed_seed=7)
    statistics = selected_residual_statistics(model, x, indices, codebooks)
    active_stages = random.Random(7).randrange(3) + 1 if dropout else 3
    assert len(statistics) == active_stages
    if shared_codebook:
        expected_count, expected_sum = initial_counts[0], initial_sums[0]
        for count, vector_sum in statistics:
            expected_count = 0.6 * expected_count + 0.4 * count
            expected_sum = 0.6 * expected_sum + 0.4 * vector_sum
        torch.testing.assert_close(
            model.layers[0]._codebook.cluster_size, expected_count
        )
        torch.testing.assert_close(model.layers[0]._codebook.embed_avg, expected_sum)
    else:
        for stage, layer in enumerate(model.layers):
            if stage < active_stages:
                count, vector_sum = statistics[stage]
                expected_count = 0.6 * initial_counts[stage] + 0.4 * count
                expected_sum = 0.6 * initial_sums[stage] + 0.4 * vector_sum
            else:
                expected_count, expected_sum = (
                    initial_counts[stage],
                    initial_sums[stage],
                )
            torch.testing.assert_close(layer._codebook.cluster_size, expected_count)
            torch.testing.assert_close(layer._codebook.embed_avg, expected_sum)


@pytest.mark.parametrize("beam_size", [None, 1, 2, 3])
def test_beam_inference_and_single_beam_controls(beam_size):
    torch.manual_seed(40)
    model = ResidualVQ(
        dim=2,
        codebook_size=4,
        num_quantizers=3,
        beam_size=beam_size,
        kmeans_init=False,
        threshold_ema_dead_code=0,
    ).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    x = torch.randn(2, 5, 2)
    quantized, indices, _ = model(x)
    reconstructed = model.get_output_from_indices(indices)
    torch.testing.assert_close(quantized, reconstructed)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("shared_codebook", [False, True])
@pytest.mark.parametrize("beam_size", [1, 2, 3])
def test_ema_update_preserves_first_call_values_and_input_gradients(
    shared_codebook, beam_size
):
    torch.manual_seed(40)
    kwargs = {
        "dim": 2,
        "codebook_size": 4,
        "num_quantizers": 3,
        "beam_size": beam_size,
        "shared_codebook": shared_codebook,
        "kmeans_init": False,
        "threshold_ema_dead_code": 0,
        "rotation_trick": False,
    }
    updating = ResidualVQ(**kwargs).train()
    fixed = ResidualVQ(**kwargs, ema_update=False).train()
    fixed.load_state_dict(updating.state_dict())
    x = torch.randn(2, 5, 2, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    quantized, indices, loss = updating(x)
    reference_quantized, reference_indices, reference_loss = fixed(reference_x)
    assert torch.equal(indices, reference_indices)
    torch.testing.assert_close(quantized, reference_quantized, rtol=0, atol=0)
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    (quantized.square().sum() + loss.sum()).backward()
    (reference_quantized.square().sum() + reference_loss.sum()).backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)


@pytest.mark.parametrize("shared_codebook", [False, True])
@pytest.mark.parametrize("use_cosine_sim", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_expired_beam_codes_use_the_selected_stage_residuals(
    shared_codebook, use_cosine_sim, masked
):
    torch.manual_seed(40)
    model = ResidualVQ(
        dim=2,
        codebook_size=4,
        num_quantizers=3,
        beam_size=2,
        decay=0.6,
        shared_codebook=shared_codebook,
        use_cosine_sim=use_cosine_sim,
        kmeans_init=False,
        threshold_ema_dead_code=100,
        rotation_trick=False,
    ).train()
    x = torch.randn(2, 5, 2)
    mask = (
        torch.tensor(
            [[True, True, False, True, False], [True, False, True, True, True]]
        )
        if masked
        else None
    )
    codebooks = [layer._codebook.embed.detach().clone() for layer in model.layers]
    _, indices, _ = model(x, mask=mask)
    residual = x.detach()
    candidates_by_stage = []
    for stage, codebook in enumerate(codebooks):
        vectors = F.normalize(residual, dim=-1) if use_cosine_sim else residual
        candidates_by_stage.append(vectors[mask] if masked else vectors.reshape(-1, 2))
        codes = F.embedding(indices[..., stage].clamp_min(0), codebook[0])
        if masked:
            codes = codes.masked_fill(~mask[..., None], 0)
        residual = residual - codes

    for layer, stage_candidates in zip(model.layers, candidates_by_stage):
        candidates = (
            torch.cat(candidates_by_stage) if shared_codebook else stage_candidates
        )
        # Every forcibly expired code must come from a valid winner-beam residual.
        distance = (
            torch.cdist(layer._codebook.embed.squeeze(0), candidates).min(-1).values
        )
        torch.testing.assert_close(
            distance, torch.zeros_like(distance), rtol=0, atol=1e-5
        )
