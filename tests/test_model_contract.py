from __future__ import annotations

import pytest
import torch

from cg_pipeline.interaction import HEInteractionBlock, InteractionContractError
from cg_pipeline.model import FixedHEClassifier
from cg_pipeline.pooling import PoolingContractError, SupportAlignedPool, flatten_pooled


def _valid_mask(size: int = 110) -> torch.Tensor:
    mask = torch.zeros((1, 1, size, size), dtype=torch.bool)
    mask[..., 52 : size - 52, 52 : size - 52] = True
    return mask


def test_interaction_exact_feature_order_masks_and_parameter_budget() -> None:
    block = HEInteractionBlock()
    h = torch.full((1, 4, 8, 110, 110), 2.0, dtype=torch.float32)
    e = torch.full_like(h, 3.0)
    valid = _valid_mask()

    output = block(h, e, valid)

    assert output.features.shape == (1, 4, 8, 7, 110, 110)
    assert output.feature_names == (
        "H_gated",
        "E_gated",
        "HE_product",
        "H_x_E_ring",
        "E_x_H_ring",
        "H_excess",
        "E_excess",
    )
    center = output.features[0, 0, 0, :, 55, 55]
    assert torch.equal(center, torch.tensor([2.0, 3.0, 6.0, 6.0, 6.0, 0.0, 1.0]))
    assert torch.equal(output.valid_support_mask, valid)
    expected_neighborhood = torch.zeros_like(valid)
    expected_neighborhood[..., 53:57, 53:57] = True
    assert torch.equal(output.neighborhood_valid_support_mask, expected_neighborhood)
    assert sum(parameter.numel() for parameter in block.parameters()) == 64
    assert all(torch.equal(parameter, torch.zeros_like(parameter)) for parameter in block.parameters())
    assert block.trainable_parameter_names == ("gate_scale", "gate_bias")


def test_interaction_is_he_exchange_equivariant_and_rejects_nonfinite() -> None:
    torch.manual_seed(11)
    block = HEInteractionBlock()
    h = torch.rand((1, 4, 8, 110, 110), dtype=torch.float32)
    e = torch.rand_like(h)
    valid = _valid_mask()

    original = block(h, e, valid).features
    swapped = block(e, h, valid).features

    assert torch.equal(swapped, original[:, :, :, [1, 0, 2, 4, 3, 6, 5]])
    h[..., 0, 0] = float("nan")
    with pytest.raises(InteractionContractError, match="non-finite"):
        block(h, e, valid)


def test_support_aligned_pooling_shape_counts_zero_variance_and_backward() -> None:
    features = torch.arange(1, 8, dtype=torch.float32).view(1, 1, 1, 7, 1, 1)
    features = features.expand(1, 4, 8, 7, 110, 110).clone().requires_grad_(True)
    valid = _valid_mask()
    neighborhood = torch.zeros_like(valid)
    neighborhood[..., 53:57, 53:57] = True

    output = SupportAlignedPool()(features, valid, neighborhood)

    assert output.statistics_float64.shape == (1, 4, 8, 7, 21, 2)
    assert output.pool_float32.shape == (1, 4, 8, 7, 21, 2)
    assert output.pool_float32.dtype == torch.float32
    assert output.valid_count.dtype == torch.int64
    assert output.valid_count.tolist() == [[16, 4, 4, 4, 4] + [1] * 16]
    assert torch.equal(output.pool_float32[..., 0], features[..., 0, 0, None].expand_as(output.pool_float32[..., 0]))
    assert torch.equal(output.pool_float32[..., 1], torch.zeros_like(output.pool_float32[..., 1]))
    output.statistics_float64[..., 1].sum().backward()
    assert features.grad is not None
    assert torch.equal(features.grad, torch.zeros_like(features.grad))


def test_support_aligned_pooling_batch_32_forward_and_backward() -> None:
    torch.manual_seed(29)
    feature_values = torch.rand((32, 1, 1, 1, 110, 110), dtype=torch.float32, requires_grad=True)
    features = feature_values.expand(-1, 4, 8, 7, -1, -1)
    valid = _valid_mask().expand(32, -1, -1, -1)
    neighborhood = torch.zeros_like(valid)
    neighborhood[..., 53:57, 53:57] = True

    output = SupportAlignedPool()(features, valid, neighborhood)

    assert output.statistics_float64.shape == (32, 4, 8, 7, 21, 2)
    assert output.pool_float32.shape == (32, 4, 8, 7, 21, 2)
    assert output.pool_float32.dtype == torch.float32
    assert torch.isfinite(output.pool_float32).all()
    output.pool_float32.sum().backward()
    assert feature_values.grad is not None
    assert torch.isfinite(feature_values.grad).all()


def test_support_aligned_pooling_dense_regions_do_not_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_values = torch.arange(110 * 110, dtype=torch.float32).reshape(
        1, 1, 1, 1, 110, 110
    )
    features = feature_values.expand(2, 4, 8, 7, -1, -1)
    valid = _valid_mask().expand(2, -1, -1, -1)
    neighborhood = torch.zeros_like(valid)
    neighborhood[..., 53:57, 53:57] = True

    def reject_sort(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("dense support must not require sorting")

    monkeypatch.setattr(torch, "argsort", reject_sort)

    output = SupportAlignedPool()(features, valid, neighborhood)

    assert output.valid_count.tolist() == [[16, 4, 4, 4, 4] + [1] * 16] * 2


def test_support_aligned_pooling_dense_path_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(31)
    size = 115
    feature_values = torch.rand(
        (2, 4, 8, 7, size, size), dtype=torch.float32, requires_grad=True
    )
    reference_values = feature_values.detach().clone().requires_grad_(True)
    valid = _valid_mask(size).expand(2, -1, -1, -1)
    neighborhood = torch.zeros_like(valid)
    neighborhood[..., 53 : size - 53, 53 : size - 53] = True

    output = SupportAlignedPool()(feature_values, valid, neighborhood)
    reference_features = reference_values.reshape(2, 224, size, size)

    def reference_balanced_sum(values: torch.Tensor) -> torch.Tensor:
        level = values
        while level.shape[-1] > 1:
            pair_count = level.shape[-1] // 2
            level = level[..., : 2 * pair_count : 2] + level[..., 1 : 2 * pair_count : 2]
            if values.shape[-1] % 2:
                level = torch.cat((level, values[..., -1:]), dim=-1)
            values = level
        return level[..., 0]

    reference_regions = []
    for region in output.geometry:
        leaves = reference_features[
            :, :, region.y_start : region.y_end, region.x_start : region.x_end
        ].reshape(2, 224, -1).to(torch.float64)
        count = leaves.shape[-1]
        mean = reference_balanced_sum(leaves) / count
        centered = leaves - mean.unsqueeze(-1)
        standard_deviation = torch.sqrt(reference_balanced_sum(centered * centered) / count)
        reference_regions.append(torch.stack((mean, standard_deviation), dim=-1))
    reference = torch.stack(reference_regions, dim=2).reshape(2, 4, 8, 7, 21, 2)

    torch.testing.assert_close(output.statistics_float64, reference, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(reference)
    output.statistics_float64.backward(upstream)
    reference.backward(upstream)

    assert feature_values.grad is not None
    assert reference_values.grad is not None
    torch.testing.assert_close(
        feature_values.grad, reference_values.grad, atol=2e-4, rtol=2e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_support_aligned_pooling_dense_cpu_cuda_forward_and_backward_are_equivalent() -> None:
    torch.manual_seed(37)
    size = 115
    values = torch.rand((1, 4, 8, 7, size, size), dtype=torch.float32)
    upstream = torch.randn((1, 4, 8, 7, 21, 2), dtype=torch.float64)
    valid = _valid_mask(size)
    neighborhood = torch.zeros_like(valid)
    neighborhood[..., 53 : size - 53, 53 : size - 53] = True

    cpu_values = values.clone().requires_grad_(True)
    cpu_output = SupportAlignedPool()(cpu_values, valid, neighborhood)
    cpu_output.statistics_float64.backward(upstream)

    cuda_values = values.cuda().requires_grad_(True)
    cuda_output = SupportAlignedPool()(
        cuda_values,
        valid.cuda(),
        neighborhood.cuda(),
    )
    cuda_output.statistics_float64.backward(upstream.cuda())

    torch.testing.assert_close(
        cuda_output.statistics_float64.cpu(),
        cpu_output.statistics_float64,
        atol=float.fromhex("0x1p-48"),
        rtol=float.fromhex("0x1p-47"),
    )
    assert cpu_values.grad is not None
    assert cuda_values.grad is not None
    torch.testing.assert_close(
        cuda_values.grad.cpu(),
        cpu_values.grad,
        atol=float.fromhex("0x1p-21"),
        rtol=float.fromhex("0x1p-19"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_support_aligned_pooling_cpu_cuda_forward_and_backward_are_equivalent() -> None:
    size = 114
    feature_values = (
        torch.arange(size * size, dtype=torch.float32).reshape(1, 1, 1, 1, size, size)
        / float(size * size)
    )
    valid = _valid_mask(size)
    neighborhood = torch.zeros_like(valid)
    coordinates = torch.arange(8)
    neighborhood[..., 53:61, 53:61] = (
        coordinates[:, None] + coordinates[None, :]
    ) % 2 == 0

    cpu_values = feature_values.clone().requires_grad_(True)
    cpu_output = SupportAlignedPool()(
        cpu_values.expand(-1, 4, 8, 7, -1, -1),
        valid,
        neighborhood,
    )
    cpu_output.pool_float32.sum().backward()

    cuda_values = feature_values.cuda().requires_grad_(True)
    cuda_output = SupportAlignedPool()(
        cuda_values.expand(-1, 4, 8, 7, -1, -1),
        valid.cuda(),
        neighborhood.cuda(),
    )
    cuda_output.pool_float32.sum().backward()

    assert torch.equal(cuda_output.valid_count.cpu(), cpu_output.valid_count)
    assert torch.equal(cuda_output.pooling_support_mask.cpu(), cpu_output.pooling_support_mask)
    torch.testing.assert_close(
        cuda_output.statistics_float64.cpu(),
        cpu_output.statistics_float64,
        atol=float.fromhex("0x1p-48"),
        rtol=float.fromhex("0x1p-47"),
    )
    torch.testing.assert_close(
        cuda_output.pool_float32.cpu(),
        cpu_output.pool_float32,
        atol=float.fromhex("0x1p-22"),
        rtol=float.fromhex("0x1p-21"),
    )
    assert cpu_values.grad is not None
    assert cuda_values.grad is not None
    torch.testing.assert_close(
        cuda_values.grad.cpu(),
        cpu_values.grad,
        atol=float.fromhex("0x1p-21"),
        rtol=float.fromhex("0x1p-19"),
    )


def test_pooling_rejects_bad_geometry_subset_empty_and_nonfinite() -> None:
    pool = SupportAlignedPool()
    features = torch.zeros((1, 4, 8, 7, 110, 110), dtype=torch.float32)
    valid = _valid_mask()
    neighborhood = torch.zeros_like(valid)
    neighborhood[..., 53:57, 53:57] = True

    with pytest.raises(PoolingContractError, match="at least 110"):
        pool(features[..., :109, :], valid[..., :109, :], neighborhood[..., :109, :])
    bad_subset = neighborhood.clone()
    bad_subset[..., 0, 0] = True
    with pytest.raises(PoolingContractError, match="subset"):
        pool(features, valid, bad_subset)
    with pytest.raises(PoolingContractError, match="empty region"):
        pool(features, valid, torch.zeros_like(neighborhood))
    features[..., 55, 55] = float("inf")
    with pytest.raises(PoolingContractError, match="non-finite"):
        pool(features, valid, neighborhood)


def test_flatten_and_linear_logit_model_exact_budget_and_fixed_frontend_step() -> None:
    model = FixedHEClassifier(frontend_backend="fft")
    assert sum(parameter.numel() for parameter in model.electronic_parameters()) == 9473
    assert list(model.frontend.parameters()) == []
    assert model.classifier.weight.shape == (1, 9408)
    assert model.classifier.bias.shape == (1,)
    assert torch.equal(model.classifier.weight, torch.zeros_like(model.classifier.weight))
    assert torch.equal(model.classifier.bias, torch.zeros_like(model.classifier.bias))

    structured = torch.arange(9408, dtype=torch.float32).reshape(1, 4, 8, 7, 21, 2)
    flattened = flatten_pooled(structured)
    assert flattened.shape == (1, 9408)
    assert torch.equal(flattened, structured.reshape(1, 9408))

    before = model.frontend.fixed_state_identity()
    optimizer = torch.optim.SGD(model.electronic_parameters(), lr=0.1)
    rgb = torch.full((1, 3, 110, 110), 255, dtype=torch.uint8)
    logits = model(rgb).logits
    assert logits.shape == (1,) and logits.dtype == torch.float32
    assert torch.equal(logits, torch.zeros_like(logits))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
    optimizer.step()

    assert model.frontend.fixed_state_identity() == before
    assert model.classifier.bias.item() != 0.0
